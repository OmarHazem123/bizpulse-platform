"""Star-schema DDL.

No declared foreign keys: DuckDB restricts updates of FK-referenced rows, so referential
integrity is guaranteed by the loader (dimensions are upserted before facts, in the
same transaction) and asserted by :func:`orphan_counts` in tests.
"""

from __future__ import annotations

from typing import Final

import duckdb

DIM_DATE: Final = "dim_date"
DIM_CUSTOMERS: Final = "dim_customers"
DIM_PRODUCTS: Final = "dim_products"
FACT_TRANSACTIONS: Final = "fact_transactions"
META_RUNS: Final = "meta_pipeline_runs"
META_QUARANTINE: Final = "meta_quarantine"
META_DATASETS: Final = "meta_datasets"

TABLES: Final[tuple[str, ...]] = (
    DIM_DATE,
    DIM_CUSTOMERS,
    DIM_PRODUCTS,
    FACT_TRANSACTIONS,
    META_RUNS,
    META_QUARANTINE,
    META_DATASETS,
)

DDL: Final[tuple[str, ...]] = (
    f"""
    CREATE TABLE IF NOT EXISTS {DIM_DATE} (
        date_key      INTEGER PRIMARY KEY,      -- yyyymmdd
        full_date     DATE NOT NULL UNIQUE,
        year          SMALLINT NOT NULL,
        quarter       TINYINT NOT NULL,
        month         TINYINT NOT NULL,
        month_name    VARCHAR NOT NULL,
        year_month    VARCHAR NOT NULL,         -- 2026-03
        iso_year      SMALLINT NOT NULL,
        iso_week      TINYINT NOT NULL,
        day_of_month  TINYINT NOT NULL,
        day_of_week   TINYINT NOT NULL,         -- 1 = Monday
        day_name      VARCHAR NOT NULL,
        is_weekend    BOOLEAN NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {DIM_CUSTOMERS} (
        customer_key      BIGINT PRIMARY KEY,   -- key64(dataset|customer|customer_id)
        dataset           VARCHAR NOT NULL,
        customer_id       VARCHAR NOT NULL,
        first_seen_date   DATE NOT NULL,
        last_seen_date    DATE NOT NULL,
        first_channel     VARCHAR NOT NULL,
        region            VARCHAR NOT NULL,
        is_anonymous      BOOLEAN NOT NULL,
        updated_run_id    VARCHAR NOT NULL,
        UNIQUE (dataset, customer_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {DIM_PRODUCTS} (
        product_key       BIGINT PRIMARY KEY,   -- key64(dataset|product|product_id)
        dataset           VARCHAR NOT NULL,
        product_id        VARCHAR NOT NULL,
        product_name      VARCHAR NOT NULL,
        category          VARCHAR NOT NULL,
        first_seen_date   DATE NOT NULL,
        last_seen_date    DATE NOT NULL,
        updated_run_id    VARCHAR NOT NULL,
        UNIQUE (dataset, product_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {FACT_TRANSACTIONS} (
        transaction_key   BIGINT PRIMARY KEY,   -- key64(dataset|transaction_id)
        dataset           VARCHAR NOT NULL,
        transaction_id    VARCHAR NOT NULL,
        order_id          VARCHAR NOT NULL,
        order_ts          TIMESTAMP NOT NULL,   -- naive UTC
        date_key          INTEGER NOT NULL,
        customer_key      BIGINT NOT NULL,
        product_key       BIGINT NOT NULL,
        channel           VARCHAR NOT NULL,
        region            VARCHAR NOT NULL,
        status            VARCHAR NOT NULL,     -- completed | refunded | cancelled
        is_refund         BOOLEAN NOT NULL,
        quantity          DOUBLE NOT NULL,
        unit_price        DOUBLE NOT NULL,      -- reporting currency
        gross_amount      DOUBLE NOT NULL,
        discount_amount   DOUBLE NOT NULL,
        net_amount        DOUBLE NOT NULL,
        cost_amount       DOUBLE,               -- NULL when unit cost unknown
        fee_amount        DOUBLE NOT NULL,
        currency_original VARCHAR NOT NULL,
        fx_rate           DOUBLE NOT NULL,
        row_hash          VARCHAR NOT NULL,     -- full SHA-256 of normalized content
        run_id            VARCHAR NOT NULL,
        loaded_at         TIMESTAMP NOT NULL,
        updated_at        TIMESTAMP NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {META_RUNS} (
        run_id            VARCHAR PRIMARY KEY,
        dataset           VARCHAR NOT NULL,
        source_file       VARCHAR NOT NULL,
        source_format     VARCHAR,
        file_sha256       VARCHAR NOT NULL,
        status            VARCHAR NOT NULL,
        started_at        TIMESTAMP NOT NULL,
        finished_at       TIMESTAMP,
        duration_seconds  DOUBLE,
        rows_read         BIGINT,
        rows_valid        BIGINT,
        rows_quarantined  BIGINT,
        rows_inserted     BIGINT,
        rows_updated      BIGINT,
        rows_unchanged    BIGINT,
        pass_rate         DOUBLE,
        mapping_json      VARCHAR,
        report_path       VARCHAR,
        error_message     VARCHAR,
        code_version      VARCHAR NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {META_QUARANTINE} (
        run_id            VARCHAR NOT NULL,
        dataset           VARCHAR NOT NULL,
        source_row        BIGINT NOT NULL,
        reason_codes      VARCHAR NOT NULL,     -- comma separated, sorted
        issues_json       VARCHAR NOT NULL,
        raw_json          VARCHAR NOT NULL,
        PRIMARY KEY (run_id, source_row)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {META_DATASETS} (
        dataset             VARCHAR PRIMARY KEY,
        display_name        VARCHAR NOT NULL,
        domain              VARCHAR NOT NULL,
        reporting_currency  VARCHAR NOT NULL,
        mapping_json        VARCHAR NOT NULL,
        updated_at          TIMESTAMP NOT NULL
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_runs_dataset_hash ON {META_RUNS} (dataset, file_sha256)",
)


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    for statement in DDL:
        con.execute(statement)


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in TABLES:
        row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
        counts[table] = int(row[0]) if row else 0
    return counts


def orphan_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Fact rows whose dimension keys are missing (must always be zero)."""
    checks = {
        "customer_key": f"""SELECT count(*) FROM {FACT_TRANSACTIONS} f
            LEFT JOIN {DIM_CUSTOMERS} d USING (customer_key) WHERE d.customer_key IS NULL""",
        "product_key": f"""SELECT count(*) FROM {FACT_TRANSACTIONS} f
            LEFT JOIN {DIM_PRODUCTS} d USING (product_key) WHERE d.product_key IS NULL""",
        "date_key": f"""SELECT count(*) FROM {FACT_TRANSACTIONS} f
            LEFT JOIN {DIM_DATE} d USING (date_key) WHERE d.date_key IS NULL""",
    }
    result: dict[str, int] = {}
    for name, sql in checks.items():
        row = con.execute(sql).fetchone()
        result[name] = int(row[0]) if row else 0
    return result
