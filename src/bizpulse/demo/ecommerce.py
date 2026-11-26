"""Northwind Commerce: a multi-currency online store exported as CSV.

Simulation: new customers arrive daily (Poisson x trend x weekly x annual seasonality x
promotional events). Every customer then repeats purchases with a churn hazard that
decays with each order and depends on a latent loyalty level, which produces realistic
cohort retention curves. Black Friday / Cyber Monday pull repeat purchases forward.
Products follow a Zipf popularity curve (Pareto 80/20).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bizpulse.demo.common import (
    Anomaly,
    DemoTable,
    hour_weights,
    seasonal_multiplier,
    window_days,
)
from bizpulse.ingestion.fx import FxTable

BASE_NEW_CUSTOMERS_PER_DAY = 19.0
WEEKLY = (0.97, 0.95, 0.96, 0.99, 1.04, 1.06, 1.03)  # Monday .. Sunday
MONTHLY = {1: 0.86, 2: 0.84, 3: 0.93, 4: 0.95, 5: 0.98, 6: 0.97,
           7: 1.0, 8: 0.98, 9: 1.0, 10: 1.04, 11: 1.32, 12: 1.42}  # fmt: skip
EVENTS = {
    "2025-11-28": 3.2,  # Black Friday
    "2025-11-29": 1.8,
    "2025-11-30": 1.5,
    "2025-12-01": 2.6,  # Cyber Monday
    "2026-07-15": 1.7,  # summer sale
}
PULL_IN_DAYS = ("2025-11-28", "2025-12-01", "2026-07-15")
BFCM_WINDOW = ("2025-11-27", "2025-12-02")
PRICE_INCREASE_FROM = pd.Timestamp("2026-01-01")

CHANNELS = ("Organic Search", "Paid Search", "Social", "Email", "Marketplace")
CHANNEL_P = (0.34, 0.26, 0.18, 0.12, 0.10)
REGIONS = ("US", "CA", "EU", "UK")
REGION_P = (0.60, 0.08, 0.18, 0.14)
REGION_CURRENCY = {"US": "USD", "CA": "CAD", "EU": "EUR", "UK": "GBP"}
CURRENCY_SYMBOL = {"USD": "$", "CAD": "C$", "EUR": "€", "GBP": "£"}

HOURS = {0: 0.4, 1: 0.25, 2: 0.15, 3: 0.1, 4: 0.1, 5: 0.2, 6: 0.5, 7: 0.9, 8: 1.2,
         9: 1.4, 10: 1.5, 11: 1.6, 12: 1.8, 13: 1.7, 14: 1.5, 15: 1.4, 16: 1.4,
         17: 1.5, 18: 1.8, 19: 2.1, 20: 2.3, 21: 2.1, 22: 1.5, 23: 0.8}  # fmt: skip

# category: (code, price log-mean, price log-sd, gross margin, refund probability, items)
CATALOG: dict[str, tuple[str, float, float, float, float, tuple[str, ...]]] = {
    "Electronics": ("ELE", 4.5, 0.55, 0.22, 0.05, (
        "Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "USB-C Hub",
        "Mechanical Keyboard", "Gaming Mouse", "4K Webcam", "Power Bank",
        "Noise-Cancelling Headphones", "Portable SSD", "Fitness Tracker", "Smart Plug",
        "LED Desk Lamp", "Phone Stand", "Wireless Charger", "Action Camera", "E-Reader",
        "Mini Projector")),
    "Home & Kitchen": ("HOM", 3.6, 0.5, 0.42, 0.03, (
        "Chef Knife", "Cast Iron Skillet", "French Press", "Bamboo Cutting Board",
        "Ceramic Mug Set", "Air Fryer", "Blender", "Throw Blanket", "Scented Candle",
        "Linen Napkins", "Spice Rack", "Glass Food Containers", "Pour-Over Kettle",
        "Salad Spinner", "Waffle Maker", "Table Runner", "Storage Baskets", "Dish Rack",
        "Knife Sharpener", "Coffee Grinder", "Toaster", "Herb Planter", "Bath Towel Set",
        "Pillow Covers")),
    "Apparel": ("APP", 3.4, 0.4, 0.55, 0.08, (
        "T-Shirt", "Hoodie", "Denim Jacket", "Chinos", "Running Shorts", "Wool Sweater",
        "Rain Jacket", "Leggings", "Polo Shirt", "Flannel Shirt", "Beanie", "Scarf",
        "Socks 3-Pack", "Sneakers", "Ankle Boots", "Linen Shirt", "Cargo Pants",
        "Puffer Vest", "Tank Top", "Sweatpants", "Baseball Cap", "Belt", "Swim Shorts",
        "Midi Skirt", "Cardigan", "Joggers", "Windbreaker", "Gloves")),
    "Beauty": ("BEA", 3.0, 0.4, 0.62, 0.025, (
        "Vitamin C Serum", "Hyaluronic Moisturizer", "SPF 50 Sunscreen", "Clay Mask",
        "Lip Balm Set", "Hair Oil", "Dry Shampoo", "Face Cleanser", "Eye Cream",
        "Body Lotion", "Nail Polish Set", "Makeup Brushes", "Retinol Night Cream",
        "Bath Bombs", "Hand Cream", "Shampoo Bar", "Toner", "Micellar Water",
        "Sheet Masks", "Viral Glow Serum")),
    "Sports": ("SPO", 3.8, 0.5, 0.40, 0.035, (
        "Yoga Mat", "Resistance Bands", "Dumbbell Pair", "Jump Rope", "Water Bottle",
        "Foam Roller", "Cycling Gloves", "Tennis Balls", "Gym Bag", "Running Belt",
        "Kettlebell", "Pull-Up Bar", "Swim Goggles", "Hiking Poles", "Camping Lantern",
        "Climbing Chalk")),
    "Books": ("BOO", 2.8, 0.3, 0.35, 0.02, (
        "Data Science Handbook", "Startup Playbook", "Mystery Novel", "Cookbook",
        "Travel Guide", "Sci-Fi Anthology", "Biography", "Children's Picture Book",
        "Productivity Guide", "Poetry Collection", "History of Trade", "Design Principles",
        "Finance Basics", "Graphic Novel")),
}  # fmt: skip
VIRAL_PRODUCT = "Viral Glow Serum"

OUTAGE = ("2026-03-10", "2026-03-12")
VIRAL_DAY = "2026-05-22"
REFUND_SPIKE = ("2026-07-14", "2026-07-20")

COLUMNS: dict[str, str | tuple[str, ...]] = {
    "order_id": "Order Number",
    "transaction_id": "Line ID",
    "order_date": "Order Date",
    "customer_id": "Customer ID",
    "product_id": "SKU",
    "product_name": "Product Title",
    "category": "Product Type",
    "quantity": "Qty",
    "unit_price": "Unit Price",
    "discount_amount": "Discount",
    "unit_cost": "Unit Cost",
    "fee_amount": "Processing Fee",
    "currency": "Currency",
    "channel": "Sales Channel",
    "region": "Ship Region",
    "status": "Financial Status",
}


def _catalog(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, (code, mu, sigma, margin, refund, items) in CATALOG.items():
        for i, name in enumerate(items, start=1):
            price = max(4.99, round(float(np.exp(rng.normal(mu, sigma)))) - 0.01)
            rows.append(
                {
                    "sku": f"{code}-{i:03d}",
                    "name": name,
                    "category": category,
                    "price_usd": price,
                    "margin": float(np.clip(margin + rng.normal(0, 0.05), 0.08, 0.85)),
                    "refund_p": refund,
                }
            )
    catalog = pd.DataFrame(rows)
    ranks = rng.permutation(len(catalog)) + 1
    weights = 1.0 / ranks.astype(np.float64) ** 1.15
    weights[catalog["name"].eq(VIRAL_PRODUCT).to_numpy()] = float(np.min(weights)) / 4
    catalog["weight"] = weights / weights.sum()
    return catalog


def _day_index(days: pd.DatetimeIndex, day: str) -> int:
    return int(days.searchsorted(pd.Timestamp(day)))


def _between(ts: pd.Series[pd.Timestamp], window: tuple[str, str]) -> pd.Series[bool]:
    day = ts.dt.normalize()
    return (day >= pd.Timestamp(window[0])) & (day <= pd.Timestamp(window[1]))


def generate(rng: np.random.Generator, scale: float, fx: FxTable) -> DemoTable:
    days = window_days()
    n_days = len(days)
    mult = seasonal_multiplier(
        days, weekly=WEEKLY, monthly=MONTHLY, monthly_growth=0.025, events=EVENTS
    )
    catalog = _catalog(rng)

    # --- customers & their order days -------------------------------------
    new_per_day = rng.poisson(BASE_NEW_CUSTOMERS_PER_DAY * scale * mult)
    acq_day = np.repeat(np.arange(n_days), new_per_day)
    n_customers = len(acq_day)
    channel = rng.choice(len(CHANNELS), size=n_customers, p=CHANNEL_P)
    region = rng.choice(len(REGIONS), size=n_customers, p=REGION_P)
    loyalty = rng.gamma(1.2, 1.0, size=n_customers)
    pull_in = [_day_index(days, d) for d in PULL_IN_DAYS]

    order_day: list[int] = []
    order_cust: list[int] = []
    for cust in range(n_customers):
        day = int(acq_day[cust])
        k = 0
        while day < n_days:
            order_day.append(day)
            order_cust.append(cust)
            k += 1
            churn = 0.62 * 0.8 ** (k - 1) / (0.6 + 0.4 * loyalty[cust])
            if rng.random() < min(0.9, max(0.12, churn)):
                break
            nxt = day + 3 + int(rng.exponential(42.0 / (0.5 + loyalty[cust])))
            for event in pull_in:
                if day < event <= nxt and rng.random() < 0.45:
                    nxt = event
                    break
            day = nxt

    orders = pd.DataFrame({"day": order_day, "cust": order_cust})
    orders["channel"] = channel[orders["cust"]]
    orders["region"] = region[orders["cust"]]
    orders["viral"] = False

    # --- anomaly: checkout outage (orders -85 %) -------------------------
    when = days[orders["day"].to_numpy()]
    outage = (when >= pd.Timestamp(OUTAGE[0])) & (when <= pd.Timestamp(OUTAGE[1]))
    dropped = outage & (rng.random(len(orders)) < 0.85)
    orders = orders.loc[~dropped].reset_index(drop=True)

    # --- anomaly: social viral spike (one product, one day) ---------------
    viral_idx = _day_index(days, VIRAL_DAY)
    recent = orders["day"].between(viral_idx - 14, viral_idx - 1)
    n_viral = int(2.5 * recent.sum() / 14)
    viral = pd.DataFrame(
        {
            "day": viral_idx,
            "cust": np.arange(n_customers, n_customers + n_viral),
            "channel": CHANNELS.index("Social"),
            "region": rng.choice(len(REGIONS), size=n_viral, p=REGION_P),
            "viral": True,
        }
    )
    orders = pd.concat([orders, viral], ignore_index=True)

    # --- timestamps & order numbers ---------------------------------------
    hours, hour_p = hour_weights(HOURS)
    n_orders = len(orders)
    offsets = (
        pd.to_timedelta(rng.choice(hours, size=n_orders, p=hour_p), unit="h")
        + pd.to_timedelta(rng.integers(0, 60, size=n_orders), unit="m")
        + pd.to_timedelta(rng.integers(0, 60, size=n_orders), unit="s")
    )
    orders["ts"] = days[orders["day"].to_numpy()] + offsets
    orders = orders.sort_values(["ts", "cust"], kind="stable").reset_index(drop=True)
    orders["order_no"] = [f"NW-{100000 + i}" for i in range(n_orders)]
    orders["cancelled"] = rng.random(n_orders) < 0.015
    in_bfcm = _between(orders["ts"], BFCM_WINDOW).to_numpy()
    promo_draw = rng.random(n_orders)
    discount_pct = np.where(
        in_bfcm,
        np.where(promo_draw < 0.65, rng.uniform(0.20, 0.30, n_orders), 0.0),
        np.where(promo_draw < 0.18, rng.uniform(0.05, 0.20, n_orders), 0.0),
    )
    orders["discount_pct"] = discount_pct

    # --- order lines ----------------------------------------------------------
    n_lines = np.minimum(rng.geometric(0.62, size=n_orders), 5)
    n_lines[orders["viral"].to_numpy()] = 1
    line_order = np.repeat(np.arange(n_orders), n_lines)
    starts = np.repeat(np.cumsum(n_lines) - n_lines, n_lines)
    line_no = np.arange(len(line_order)) - starts + 1
    sku_idx = rng.choice(len(catalog), size=len(line_order), p=catalog["weight"].to_numpy())
    viral_sku = int(np.flatnonzero(catalog["name"].eq(VIRAL_PRODUCT).to_numpy())[0])
    sku_idx[orders["viral"].to_numpy()[line_order]] = viral_sku
    qty = rng.choice([1, 2, 3], size=len(line_order), p=[0.78, 0.17, 0.05])

    lines = orders.iloc[line_order].reset_index(drop=True)
    lines["line_no"] = line_no
    lines["n_lines"] = n_lines[line_order]
    prod = catalog.iloc[sku_idx].reset_index(drop=True)
    currency = lines["region"].map(lambda r: REGION_CURRENCY[REGIONS[r]])
    rate = currency.map(lambda c: fx.rates_to_usd[c]).to_numpy()
    uplift = np.where(lines["ts"] >= PRICE_INCREASE_FROM, 1.05, 1.0)
    price_usd = np.maximum(4.99, np.round(prod["price_usd"].to_numpy() * uplift) - 0.01)
    unit_price = np.round(price_usd / rate, 2)
    unit_cost = np.round(price_usd * (1 - prod["margin"].to_numpy()) / rate, 2)
    gross = qty * unit_price
    discount = np.round(gross * lines["discount_pct"].to_numpy(), 2)
    fee = np.round(0.029 * (gross - discount) + 0.30 / lines["n_lines"].to_numpy(), 2)

    refund_p = prod["refund_p"].to_numpy().copy()
    spike = (
        _between(lines["ts"], REFUND_SPIKE).to_numpy() & prod["category"].eq("Apparel").to_numpy()
    )
    refund_p[spike] = 0.45
    refunded = rng.random(len(lines)) < refund_p
    status = np.where(
        lines["cancelled"].to_numpy(), "voided", np.where(refunded, "refunded", "paid")
    )

    # --- messy-but-valid formatting ---------------------------------------
    n = len(lines)
    draw = rng.random((6, n))
    ts = lines["ts"]
    iso = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    us = ts.dt.strftime("%m/%d/%Y %I:%M %p")
    iso_z = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    order_date = np.where(draw[0] < 0.88, iso, np.where(draw[0] < 0.95, us, iso_z))

    plain = [f"{p:.2f}" for p in unit_price]
    symbol_style = [_with_symbol(p, c) for p, c in zip(unit_price, currency, strict=True)]
    code_style = [f"{c} {p:.2f}" for p, c in zip(unit_price, currency, strict=True)]
    price_text = np.where(draw[1] < 0.80, plain, np.where(draw[1] < 0.92, symbol_style, code_style))
    has_hint = draw[1] >= 0.80
    currency_text = np.where(has_hint & (draw[2] < 0.25), "", currency.to_numpy())

    customer_id = np.array([f"C{c + 100000}" for c in lines["cust"]], dtype=object)
    padded = draw[3] < 0.02
    customer_id[padded] = [f"  {c} " for c in customer_id[padded]]
    status_text = status.astype(object)
    paid = status == "paid"
    status_text[paid & (draw[4] < 0.05)] = "Paid "
    status_text[paid & (draw[4] >= 0.05) & (draw[4] < 0.10)] = "PAID"
    cost_text = np.array([f"{c:.2f}" for c in unit_cost], dtype=object)
    blank_cost = draw[5] < 0.015
    cost_text[blank_cost] = ""

    frame = pd.DataFrame(
        {
            "Order Number": lines["order_no"].to_numpy(),
            "Line ID": [f"{o}-{n_}" for o, n_ in zip(lines["order_no"], line_no, strict=True)],
            "Order Date": order_date,
            "Customer ID": customer_id,
            "SKU": prod["sku"].to_numpy(),
            "Product Title": prod["name"].to_numpy(),
            "Product Type": prod["category"].to_numpy(),
            "Qty": [str(q) for q in qty],
            "Unit Price": price_text,
            "Discount": [f"{d:.2f}" for d in discount],
            "Unit Cost": cost_text,
            "Processing Fee": [f"{f:.2f}" for f in fee],
            "Currency": currency_text,
            "Sales Channel": [CHANNELS[c] for c in lines["channel"]],
            "Ship Region": [REGIONS[r] for r in lines["region"]],
            "Financial Status": status_text,
        }
    ).astype(object)

    anomalies = [
        Anomaly(
            "checkout_outage", "drop", "net_revenue", OUTAGE[0], OUTAGE[1], "CRITICAL",
            "Payment provider outage: ~85 % of orders failed for three days.",
        ),
        Anomaly(
            "social_viral_spike", "spike", "net_revenue", VIRAL_DAY, VIRAL_DAY, "CRITICAL",
            f"A social video made '{VIRAL_PRODUCT}' go viral for one day.",
            {"channel": "social", "category": "Beauty"},
        ),
        Anomaly(
            "apparel_refund_spike", "rate_spike", "refund_rate", REFUND_SPIKE[0],
            REFUND_SPIKE[1], "WARNING",
            "Defective apparel batch: 45 % of Apparel lines refunded for a week.",
            {"category": "Apparel"},
        ),
    ]  # fmt: skip
    messy = {
        "us_format_dates": int(((draw[0] >= 0.88) & (draw[0] < 0.95)).sum()),
        "iso_utc_dates": int((draw[0] >= 0.95).sum()),
        "symbol_prices": int(((draw[1] >= 0.80) & (draw[1] < 0.92)).sum()),
        "code_prices": int((draw[1] >= 0.92).sum()),
        "currency_from_price_hint": int((has_hint & (draw[2] < 0.25)).sum()),
        "padded_customer_ids": int(padded.sum()),
        "status_case_variants": int((paid & (draw[4] < 0.10)).sum()),
        "unknown_unit_cost": int(blank_cost.sum()),
    }
    return DemoTable(
        dataset="ecommerce",
        frame=frame,
        columns=COLUMNS,
        gross=gross.astype(np.float64),
        anomalies=anomalies,
        messy_valid=messy,
    )


def _with_symbol(price: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOL[currency]
    if currency == "EUR":
        return f"{price:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{symbol}{price:,.2f}"
