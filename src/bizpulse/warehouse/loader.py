"""Idempotent, transactional loading of a staged fact batch into the star schema.

Strategy (per table): stage into a TEMP table, then
``UPDATE ... FROM staged`` for existing keys whose content changed, and an anti-join
``INSERT`` for new keys. No delete/re-insert of primary keys, so DuckDB's in-transaction
uniqueness checks never trip, and insert/update/unchanged counts are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb
import pandas as pd

from bizpulse.warehouse.schema import DIM_CUSTOMERS, DIM_DATE, DIM_PRODUCTS, FACT_TRANSACTIONS
from bizpulse.warehouse.transform import FACT_COLUMNS, STAGING_EXTRA


class LoadError(RuntimeError):
    """The batch violates a loader precondition."""


@dataclass(frozen=True)
class LoadResult:
    inserted: int
    updated: int
    unchanged: int
    customers_new: int
    products_new: int
    dates_new: int


_FACT_UPDATE_COLUMNS = [c for c in FACT_COLUMNS if c != "transaction_key"]


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _upsert_dates(con: duckdb.DuckDBPyConnection) -> int:
    before = _scalar(con, f"SELECT count(*) FROM {DIM_DATE}")
    con.execute(
        f"""
        INSERT INTO {DIM_DATE}
        SELECT
            CAST(strftime(d, '%Y%m%d') AS INTEGER),
            d, year(d), quarter(d), month(d), monthname(d), strftime(d, '%Y-%m'),
            isoyear(d), weekofyear(d), day(d), isodow(d), dayname(d), isodow(d) >= 6
        FROM (
            SELECT CAST(gs AS DATE) AS d
            FROM generate_series(
                (SELECT min(order_ts)::DATE FROM stg_facts),
                (SELECT max(order_ts)::DATE FROM stg_facts),
                INTERVAL 1 DAY
            ) AS t(gs)
        ) days
        WHERE NOT EXISTS (SELECT 1 FROM {DIM_DATE} x WHERE x.full_date = days.d)
        """
    )
    return _scalar(con, f"SELECT count(*) FROM {DIM_DATE}") - before


def _upsert_customers(con: duckdb.DuckDBPyConnection, run_id: str) -> int:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE stg_customers AS
        SELECT customer_key,
               any_value(dataset)               AS dataset,
               any_value(customer_id)           AS customer_id,
               min(order_ts)::DATE              AS first_seen,
               max(order_ts)::DATE              AS last_seen,
               arg_min(channel, order_ts)       AS first_channel,
               arg_max(region, order_ts)        AS region,
               bool_or(is_anonymous)            AS is_anonymous
        FROM stg_facts GROUP BY customer_key
        """
    )
    con.execute(
        f"""
        UPDATE {DIM_CUSTOMERS} AS d SET
            first_channel   = CASE WHEN s.first_seen < d.first_seen_date
                                   THEN s.first_channel ELSE d.first_channel END,
            region          = CASE WHEN s.last_seen >= d.last_seen_date
                                   THEN s.region ELSE d.region END,
            first_seen_date = least(d.first_seen_date, s.first_seen),
            last_seen_date  = greatest(d.last_seen_date, s.last_seen),
            is_anonymous    = s.is_anonymous,
            updated_run_id  = ?
        FROM stg_customers AS s
        WHERE d.customer_key = s.customer_key
        """,
        [run_id],
    )
    new = _scalar(
        con,
        f"""SELECT count(*) FROM stg_customers s WHERE NOT EXISTS
            (SELECT 1 FROM {DIM_CUSTOMERS} d WHERE d.customer_key = s.customer_key)""",
    )
    con.execute(
        f"""
        INSERT INTO {DIM_CUSTOMERS}
        SELECT customer_key, dataset, customer_id, first_seen, last_seen, first_channel,
               region, is_anonymous, ?
        FROM stg_customers s
        WHERE NOT EXISTS (SELECT 1 FROM {DIM_CUSTOMERS} d WHERE d.customer_key = s.customer_key)
        """,
        [run_id],
    )
    return new


def _upsert_products(con: duckdb.DuckDBPyConnection, run_id: str) -> int:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE stg_products AS
        SELECT product_key,
               any_value(dataset)               AS dataset,
               any_value(product_id)            AS product_id,
               arg_max(product_name, order_ts)  AS product_name,
               arg_max(category, order_ts)      AS category,
               min(order_ts)::DATE              AS first_seen,
               max(order_ts)::DATE              AS last_seen
        FROM stg_facts GROUP BY product_key
        """
    )
    con.execute(
        f"""
        UPDATE {DIM_PRODUCTS} AS d SET
            product_name    = CASE WHEN s.last_seen >= d.last_seen_date
                                   THEN s.product_name ELSE d.product_name END,
            category        = CASE WHEN s.last_seen >= d.last_seen_date
                                   THEN s.category ELSE d.category END,
            first_seen_date = least(d.first_seen_date, s.first_seen),
            last_seen_date  = greatest(d.last_seen_date, s.last_seen),
            updated_run_id  = ?
        FROM stg_products AS s
        WHERE d.product_key = s.product_key
        """,
        [run_id],
    )
    new = _scalar(
        con,
        f"""SELECT count(*) FROM stg_products s WHERE NOT EXISTS
            (SELECT 1 FROM {DIM_PRODUCTS} d WHERE d.product_key = s.product_key)""",
    )
    con.execute(
        f"""
        INSERT INTO {DIM_PRODUCTS}
        SELECT product_key, dataset, product_id, product_name, category, first_seen, last_seen, ?
        FROM stg_products s
        WHERE NOT EXISTS (SELECT 1 FROM {DIM_PRODUCTS} d WHERE d.product_key = s.product_key)
        """,
        [run_id],
    )
    return new


def _upsert_facts(
    con: duckdb.DuckDBPyConnection, run_id: str, loaded_at: datetime
) -> tuple[int, int, int]:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stg_actions AS
        SELECT s.transaction_key,
               CASE WHEN f.transaction_key IS NULL THEN 'insert'
                    WHEN f.row_hash <> s.row_hash THEN 'update'
                    ELSE 'unchanged' END AS action
        FROM stg_facts s LEFT JOIN {FACT_TRANSACTIONS} f USING (transaction_key)
        """
    )
    set_clause = ",\n            ".join(f"{c} = s.{c}" for c in _FACT_UPDATE_COLUMNS)
    con.execute(
        f"""
        UPDATE {FACT_TRANSACTIONS} AS f SET
            {set_clause},
            run_id = ?,
            updated_at = ?
        FROM stg_facts AS s
        WHERE f.transaction_key = s.transaction_key
          AND s.transaction_key IN (SELECT transaction_key FROM stg_actions WHERE action = 'update')
        """,
        [run_id, loaded_at],
    )
    columns = ", ".join(FACT_COLUMNS)
    select = ", ".join(f"s.{c}" for c in FACT_COLUMNS)
    con.execute(
        f"""
        INSERT INTO {FACT_TRANSACTIONS} ({columns}, run_id, loaded_at, updated_at)
        SELECT {select}, ?, ?, ?
        FROM stg_facts s JOIN stg_actions a USING (transaction_key)
        WHERE a.action = 'insert'
        """,
        [run_id, loaded_at, loaded_at],
    )
    counts = dict(
        con.execute("SELECT action, count(*) FROM stg_actions GROUP BY action").fetchall()
    )
    return (
        int(counts.get("insert", 0)),
        int(counts.get("update", 0)),
        int(counts.get("unchanged", 0)),
    )


def load_facts(
    con: duckdb.DuckDBPyConnection, facts: pd.DataFrame, *, run_id: str, loaded_at: datetime
) -> LoadResult:
    """Upsert dimensions then facts in ONE transaction (all-or-nothing)."""
    if facts.empty:
        return LoadResult(0, 0, 0, 0, 0, 0)
    if facts["transaction_key"].duplicated().any():
        raise LoadError("staged batch contains duplicate transaction keys")
    missing = [c for c in (*FACT_COLUMNS, *STAGING_EXTRA) if c not in facts.columns]
    if missing:
        raise LoadError(f"staged batch is missing columns: {missing}")
    staged = facts.loc[:, [*FACT_COLUMNS, *STAGING_EXTRA]]
    con.execute("BEGIN TRANSACTION")
    try:
        con.register("stg_facts_view", staged)
        con.execute("CREATE OR REPLACE TEMP TABLE stg_facts AS SELECT * FROM stg_facts_view")
        con.unregister("stg_facts_view")
        dates_new = _upsert_dates(con)
        customers_new = _upsert_customers(con, run_id)
        products_new = _upsert_products(con, run_id)
        inserted, updated, unchanged = _upsert_facts(con, run_id, loaded_at)
        for temp in ("stg_facts", "stg_customers", "stg_products", "stg_actions"):
            con.execute(f"DROP TABLE IF EXISTS {temp}")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return LoadResult(inserted, updated, unchanged, customers_new, products_new, dates_new)
