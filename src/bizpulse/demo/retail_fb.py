"""Bean & Crumb: a three-store cafe chain POS export (Parquet, all-string columns).

Simulation: hourly tickets per store with store-specific weekday profiles and summer
seasonality; daypart-aware baskets (bakery in the morning, sandwiches at lunch, cold
drinks in summer); loyalty members with heterogeneous visit frequency and finite
lifetimes (cohort retention) plus anonymous ``GUEST`` tickets; delivery-platform
commission recorded as a fee.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from bizpulse.demo.common import Anomaly, DemoTable, hour_weights, seasonal_multiplier, window_days
from bizpulse.ingestion.fx import FxTable

STORES = ("Downtown", "Riverside", "Airport")
STORE_CODE = {"Downtown": "DT", "Riverside": "RV", "Airport": "AP"}
STORE_BASE = {"Downtown": 26.0, "Riverside": 20.0, "Airport": 22.0}
STORE_WEEKLY = {
    "Downtown": (1.1, 1.1, 1.1, 1.1, 1.05, 0.85, 0.7),
    "Riverside": (0.85, 0.85, 0.9, 0.95, 1.05, 1.35, 1.3),
    "Airport": (1.0, 0.95, 0.95, 1.0, 1.1, 1.0, 1.1),
}
MONTHLY = {1: 0.88, 2: 0.9, 3: 0.97, 4: 1.0, 5: 1.03, 6: 1.1,
           7: 1.14, 8: 1.12, 9: 1.02, 10: 0.98, 11: 0.95, 12: 1.08}  # fmt: skip
HOURS = {7: 1.2, 8: 2.2, 9: 2.0, 10: 1.4, 11: 1.3, 12: 2.0, 13: 2.1, 14: 1.3,
         15: 1.0, 16: 1.0, 17: 0.9, 18: 0.7, 19: 0.5, 20: 0.4, 21: 0.3}  # fmt: skip
ORDER_TYPES = ("dine_in", "takeaway", "delivery")
ORDER_TYPE_P = (0.50, 0.35, 0.15)
DELIVERY_COMMISSION = 0.25
PRICE_INCREASE_FROM = pd.Timestamp("2026-02-01")

# item_code: (name, category, price EUR, food-cost ratio)
MENU: dict[str, tuple[str, str, float, float]] = {
    "COF-ESP": ("Espresso", "Coffee", 2.4, 0.18), "COF-AME": ("Americano", "Coffee", 2.9, 0.18),
    "COF-CAP": ("Cappuccino", "Coffee", 3.4, 0.22), "COF-LAT": ("Latte", "Coffee", 3.6, 0.22),
    "COF-FLW": ("Flat White", "Coffee", 3.5, 0.22), "COF-MOC": ("Mocha", "Coffee", 3.9, 0.25),
    "TEA-ENG": ("English Breakfast", "Tea", 2.6, 0.12), "TEA-GRN": ("Green Tea", "Tea", 2.7, 0.12),
    "TEA-CHA": ("Chai Latte", "Tea", 3.6, 0.24),
    "CLD-ICL": ("Iced Latte", "Cold Drinks", 3.9, 0.24),
    "CLD-CBR": ("Cold Brew", "Cold Drinks", 3.8, 0.2),
    "CLD-LEM": ("Lemonade", "Cold Drinks", 3.2, 0.2),
    "CLD-SMO": ("Berry Smoothie", "Cold Drinks", 4.8, 0.32),
    "CLD-ITE": ("Iced Tea", "Cold Drinks", 3.0, 0.15),
    "CLD-FRP": ("Caramel Frappe", "Cold Drinks", 4.6, 0.3),
    "BAK-CRO": ("Croissant", "Bakery", 2.6, 0.3),
    "BAK-PAC": ("Pain au Chocolat", "Bakery", 2.9, 0.3),
    "BAK-MUF": ("Blueberry Muffin", "Bakery", 3.1, 0.28),
    "BAK-CIN": ("Cinnamon Roll", "Bakery", 3.4, 0.27),
    "BAK-BAN": ("Banana Bread", "Bakery", 3.2, 0.25),
    "SAN-CLB": ("Club Sandwich", "Sandwiches", 7.9, 0.34),
    "SAN-CAP": ("Caprese Panini", "Sandwiches", 7.2, 0.33),
    "SAN-TUN": ("Tuna Melt", "Sandwiches", 7.5, 0.35),
    "SAN-CHP": ("Chicken Pesto Wrap", "Sandwiches", 8.2, 0.35),
    "SAN-VEG": ("Veggie Wrap", "Sandwiches", 6.9, 0.3),
    "DES-CHE": ("Cheesecake", "Desserts", 4.6, 0.3), "DES-BRO": ("Brownie", "Desserts", 3.2, 0.25),
    "DES-TIR": ("Tiramisu", "Desserts", 4.9, 0.32), "DES-COO": ("Cookie", "Desserts", 2.2, 0.2),
}  # fmt: skip
CATEGORY_CODES: dict[str, list[str]] = {}
for _code, (_name, _cat, _p, _c) in MENU.items():
    CATEGORY_CODES.setdefault(_cat, []).append(_code)

CLOSURE = ("2026-01-20", "2026-01-21", "Riverside")
BOGO_DAY = "2026-08-08"
POS_OUTAGE = ("2026-04-14", 11, 15)

COLUMNS: dict[str, str | tuple[str, ...]] = {
    "order_id": "ticket_id",
    "transaction_id": ("ticket_id", "line_no"),
    "order_date": "ts",
    "customer_id": "loyalty_id",
    "product_id": "item_code",
    "product_name": "item_name",
    "category": "menu_category",
    "quantity": "qty",
    "unit_price": "price",
    "discount_amount": "promo_discount",
    "unit_cost": "food_cost",
    "fee_amount": "platform_fee",
    "channel": "order_type",
    "region": "store",
    "status": "status",
}


def _pick(rng: np.random.Generator, category: str) -> str:
    codes = CATEGORY_CODES[category]
    return str(codes[int(rng.integers(0, len(codes)))])


def _basket(rng: np.random.Generator, hour: int, month: int, bogo: bool) -> list[str]:
    summer = month in (6, 7, 8)
    items: list[str] = []
    cold_p = 0.45 if summer else 0.15
    if bogo:
        cold_p = 0.8
    drinks = 1 + int(rng.random() < 0.15) if rng.random() < 0.92 else 0
    for _ in range(drinks):
        roll = rng.random()
        if roll < cold_p:
            items.append(_pick(rng, "Cold Drinks"))
        elif roll < cold_p + (1 - cold_p) * 0.8:
            items.append(_pick(rng, "Coffee"))
        else:
            items.append(_pick(rng, "Tea"))
    if hour < 11 and rng.random() < 0.6:
        items.append(_pick(rng, "Bakery"))
    elif 11 <= hour < 15 and rng.random() < 0.55:
        items.append(_pick(rng, "Sandwiches"))
    if hour >= 14 and rng.random() < 0.3:
        items.append(_pick(rng, "Desserts"))
    if not items:
        items.append(_pick(rng, "Bakery"))
    return items


def _members(
    rng: np.random.Generator, days: pd.DatetimeIndex, scale: float
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    signups = rng.poisson(3.2 * scale, size=len(days))
    signup_day = np.repeat(np.arange(len(days)), signups).astype(np.int64)
    n = len(signup_day)
    lifetime = (14 + rng.exponential(160.0, size=n)).astype(np.int64)
    home = rng.choice(len(STORES), size=n, p=(0.42, 0.3, 0.28)).astype(np.int64)
    weight = rng.lognormal(0.0, 0.9, size=n)
    return signup_day, signup_day + lifetime, home, weight


def generate(rng: np.random.Generator, scale: float, fx: FxTable) -> DemoTable:
    del fx  # single-currency dataset (EUR)
    days = window_days()
    hours, hour_p = hour_weights(HOURS)
    signup, churn, home, weight = _members(rng, days, scale)

    records: list[tuple[object, ...]] = []
    ticket_no = 0
    for s_idx, store in enumerate(STORES):
        mult = seasonal_multiplier(
            days, weekly=STORE_WEEKLY[store], monthly=MONTHLY, monthly_growth=0.012
        )
        bogo_mask = days == pd.Timestamp(BOGO_DAY)
        mult[bogo_mask] *= 1.6
        counts = rng.poisson(STORE_BASE[store] * scale * mult)
        for d_idx, day in enumerate(days):
            if store == CLOSURE[2] and CLOSURE[0] <= day.strftime("%Y-%m-%d") <= CLOSURE[1]:
                continue
            n_tickets = int(counts[d_idx])
            if n_tickets == 0:
                continue
            active = np.flatnonzero((signup <= d_idx) & (churn > d_idx))
            local = active[home[active] == s_idx]
            ticket_hours = rng.choice(hours, size=n_tickets, p=hour_p)
            is_member = rng.random(n_tickets) < 0.42
            for t in range(n_tickets):
                hour = int(ticket_hours[t])
                if (
                    day.strftime("%Y-%m-%d") == POS_OUTAGE[0]
                    and POS_OUTAGE[1] <= hour < POS_OUTAGE[2]
                ):
                    continue
                pool = local if (rng.random() < 0.85 and len(local)) else active
                member = -1
                if is_member[t] and len(pool):
                    w = weight[pool]
                    member = int(pool[rng.choice(len(pool), p=w / w.sum())])
                ticket_no += 1
                minute, second = int(rng.integers(0, 60)), int(rng.integers(0, 60))
                offset_s = hour * 3600 + minute * 60 + second
                order_type = str(rng.choice(ORDER_TYPES, p=ORDER_TYPE_P))
                status_roll = rng.random()
                status = (
                    "void"
                    if status_roll < 0.009
                    else "refunded"
                    if status_roll < 0.015
                    else "completed"
                )
                bogo = bool(bogo_mask[d_idx])
                for line_no, code in enumerate(_basket(rng, hour, day.month, bogo), start=1):
                    records.append(
                        (f"{STORE_CODE[store]}-{ticket_no:07d}", line_no, d_idx, offset_s,
                         member, code, order_type, store, status, bogo, float(rng.random()))
                    )  # fmt: skip

    lines = pd.DataFrame(
        records,
        columns=["ticket_id", "line_no", "day_idx", "offset_s", "member", "code", "order_type",
                 "store", "status", "bogo", "u"],
    )  # fmt: skip
    lines["ts"] = days[lines["day_idx"].to_numpy()] + pd.to_timedelta(
        lines["offset_s"].to_numpy(), unit="s"
    )
    lines = lines.sort_values(["ts", "ticket_id", "line_no"], kind="stable").reset_index(drop=True)
    menu = pd.DataFrame.from_dict(
        MENU, orient="index", columns=["name", "category", "price", "cost_ratio"]
    )
    item = menu.loc[lines["code"]].reset_index(drop=True)
    uplift = np.where(lines["ts"] >= PRICE_INCREASE_FROM, 1.06, 1.0)
    price = np.round(item["price"].to_numpy() * uplift, 2)
    qty = np.where(lines["u"] < 0.88, 1, np.where(lines["u"] < 0.98, 2, 3))
    cold = item["category"].eq("Cold Drinks").to_numpy()
    bogo_line = lines["bogo"].to_numpy() & cold
    qty = np.where(bogo_line, qty * 2, qty)
    gross = qty * price
    member_line = lines["member"].to_numpy() >= 0
    first_drink = (lines["line_no"].to_numpy() == 1) & item["category"].isin(
        ["Coffee", "Tea", "Cold Drinks"]
    ).to_numpy()
    discount = np.where(
        bogo_line, gross * 0.5, np.where(member_line & first_drink, price * 0.10, 0.0)
    )
    discount = np.round(discount, 2)
    food_cost = np.round(price * item["cost_ratio"].to_numpy(), 2)
    delivery = lines["order_type"].eq("delivery").to_numpy()
    fee = np.round(np.where(delivery, (gross - discount) * DELIVERY_COMMISSION, 0.0), 2)

    rng_fmt = rng.random((2, len(lines)))
    line_ts: pd.Series[pd.Timestamp] = lines["ts"]
    offset = np.where(line_ts.dt.month.between(4, 10), "+02:00", "+01:00")
    local_iso = line_ts.dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy().astype(object) + offset
    shift = pd.Series(np.where(offset == "+02:00", 2, 1), index=line_ts.index)
    utc: pd.Series[pd.Timestamp] = line_ts - pd.to_timedelta(shift, unit="h")
    utc_z = utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    utc_dayfirst = utc.dt.strftime("%d/%m/%Y %H:%M:%S").to_numpy()
    ts_text = np.where(
        rng_fmt[0] < 0.94, local_iso, np.where(rng_fmt[0] < 0.97, utc_z, utc_dayfirst)
    )
    plain = np.array([f"{p:.2f}" for p in price], dtype=object)
    comma = np.array([f"{p:.2f}".replace(".", ",") for p in price], dtype=object)
    euro = np.array([f"€{p:.2f}" for p in price], dtype=object)
    price_text = np.where(rng_fmt[1] < 0.93, plain, np.where(rng_fmt[1] < 0.98, comma, euro))

    frame = pd.DataFrame(
        {
            "ticket_id": lines["ticket_id"],
            "line_no": lines["line_no"].astype(str),
            "ts": ts_text,
            "loyalty_id": np.where(
                member_line, [f"L-{m + 1:05d}" for m in lines["member"]], "GUEST"
            ),
            "item_code": lines["code"],
            "item_name": item["name"].to_numpy(),
            "menu_category": item["category"].to_numpy(),
            "qty": [str(q) for q in qty],
            "price": price_text,
            "promo_discount": [f"{d:.2f}" for d in discount],
            "food_cost": [f"{c:.2f}" for c in food_cost],
            "platform_fee": [f"{f:.2f}" for f in fee],
            "order_type": lines["order_type"],
            "store": lines["store"],
            "status": lines["status"],
        }
    ).astype(object)

    anomalies = [
        Anomaly(
            "riverside_closure", "drop", "net_revenue", CLOSURE[0], CLOSURE[1], "CRITICAL",
            "Riverside store closed for two days after a burst pipe.", {"region": "Riverside"},
        ),
        Anomaly(
            "pos_outage", "drop", "net_revenue", POS_OUTAGE[0], POS_OUTAGE[0], "WARNING",
            "POS outage 11:00-15:00 across all stores (lunch peak lost).",
        ),
        Anomaly(
            "bogo_cold_drinks", "spike", "orders", BOGO_DAY, BOGO_DAY, "WARNING",
            "Buy-one-get-one cold drinks promotion (+60 % traffic, 50 % off cold drinks).",
            {"category": "Cold Drinks"},
        ),
    ]  # fmt: skip
    messy = {
        "offset_timestamps": int((rng_fmt[0] < 0.94).sum()),
        "utc_z_timestamps": int(((rng_fmt[0] >= 0.94) & (rng_fmt[0] < 0.97)).sum()),
        "dayfirst_timestamps": int((rng_fmt[0] >= 0.97).sum()),
        "decimal_comma_prices": int(((rng_fmt[1] >= 0.93) & (rng_fmt[1] < 0.98)).sum()),
        "euro_symbol_prices": int((rng_fmt[1] >= 0.98).sum()),
        "anonymous_guest_rows": int((~member_line).sum()),
    }
    return DemoTable(
        dataset="retail_fb",
        frame=frame,
        columns=COLUMNS,
        gross=gross.astype(np.float64),
        anomalies=anomalies,
        messy_valid=messy,
        has_currency_column=False,
    )
