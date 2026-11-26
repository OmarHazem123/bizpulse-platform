"""CloudLedger: a B2B SaaS billing export (Excel, native cell types).

Simulation: monthly acquisition cohorts; per-seat plans (Starter/Growth/Scale) billed
monthly or annually; churn hazard by plan that decays with tenure; seat expansion and
plan upgrades; one-off onboarding services and a monthly premium-support add-on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bizpulse.demo.common import WINDOW_END, WINDOW_START, Anomaly, DemoTable
from bizpulse.ingestion.fx import FxTable

BASE_NEW_ACCOUNTS_PER_MONTH = 58.0
MONTHLY = {1: 1.2, 2: 1.0, 3: 1.05, 4: 0.95, 5: 1.0, 6: 0.9,
           7: 0.85, 8: 0.85, 9: 1.2, 10: 1.1, 11: 1.0, 12: 0.75}  # fmt: skip
PLANS = ("Starter", "Growth", "Scale")
PLAN_P = (0.55, 0.33, 0.12)
PLAN_CODE = {"Starter": "STR", "Growth": "GRW", "Scale": "SCL"}
# per-seat monthly list prices by currency, and USD cost per seat
PRICE_BOOK = {
    "USD": {"Starter": 15.0, "Growth": 35.0, "Scale": 60.0},
    "EUR": {"Starter": 14.0, "Growth": 32.0, "Scale": 55.0},
    "GBP": {"Starter": 12.0, "Growth": 28.0, "Scale": 48.0},
}
COST_PER_SEAT_USD = {"Starter": 4.0, "Growth": 7.5, "Scale": 11.0}
CHURN = {"Starter": 0.06, "Growth": 0.035, "Scale": 0.02}
SEATS_LOGMEAN = {"Starter": (1.0, 0.5), "Growth": (2.0, 0.6), "Scale": (3.2, 0.5)}
CHANNELS = ("organic", "paid_search", "partner", "outbound_sales", "referral")
CHANNEL_P = {
    "Starter": (0.40, 0.35, 0.10, 0.03, 0.12),
    "Growth": (0.30, 0.25, 0.20, 0.10, 0.15),
    "Scale": (0.15, 0.10, 0.25, 0.40, 0.10),
}
REGIONS = ("NA", "EMEA", "APAC", "LATAM")
REGION_P = (0.50, 0.30, 0.15, 0.05)
SUPPORT_PRICE_USD, SUPPORT_COST_USD = 199.0, 80.0
ONBOARDING_PRICE_USD = {"Growth": 500.0, "Scale": 1500.0}
ONBOARDING_COST_RATIO = 0.6
STATUS = ("Paid", "Refunded", "Void", "Payment Failed")
STATUS_P = (0.955, 0.015, 0.01, 0.02)

CHURN_SPIKE_MONTH = pd.Period("2026-02", freq="M")
ENTERPRISE_DEAL_DAY = pd.Timestamp("2026-06-15 11:30")
ENTERPRISE_DEAL_USD = 48_000.0

COLUMNS: dict[str, str | tuple[str, ...]] = {
    "order_id": "Invoice No",
    "transaction_id": "Line Ref",
    "order_date": "Invoice Date",
    "customer_id": "Account ID",
    "product_id": "Plan Code",
    "product_name": "Plan Name",
    "category": "Product Family",
    "quantity": "Seats",
    "unit_price": "Price per Seat",
    "discount_amount": "Discount Amt",
    "unit_cost": "Cost per Seat",
    "fee_amount": "Payment Fee",
    "currency": "Currency",
    "channel": "Aquisition Channel",  # deliberate typo: exercises fuzzy header matching
    "region": "Region",
    "status": "Invoice Status",
}


def _local(amount_usd: float, currency: str, fx: FxTable) -> float:
    return round(amount_usd / fx.rates_to_usd[currency], 2)


def generate(rng: np.random.Generator, scale: float, fx: FxTable) -> DemoTable:
    months = pd.period_range(WINDOW_START, WINDOW_END, freq="M")
    lines: list[dict[str, object]] = []
    invoice_no = 0
    account_no = 0

    def add_invoice(
        when: pd.Timestamp,
        account: int,
        items: list[tuple[str, str, str, int, float, float, float]],
        meta: dict[str, str],
    ) -> None:
        nonlocal invoice_no
        invoice_no += 1
        status = str(rng.choice(STATUS, p=STATUS_P))
        for line, (code, name, family, qty, price, discount, cost) in enumerate(items, 1):
            net = qty * price - discount
            lines.append(
                {
                    "Invoice No": f"INV-{invoice_no:06d}",
                    "Line Ref": f"INV-{invoice_no:06d}-L{line}",
                    "Invoice Date": when,
                    "Account ID": 100000 + account,
                    "Plan Code": code,
                    "Plan Name": name,
                    "Product Family": family,
                    "Seats": qty,
                    "Price per Seat": price,
                    "Discount Amt": round(discount, 2),
                    "Cost per Seat": cost,
                    "Currency": meta["currency"],
                    "Aquisition Channel": meta["channel"],
                    "Region": meta["region"],
                    "Invoice Status": status,
                    "Payment Fee": round(0.029 * max(net, 0.0) + 0.30, 2),
                }
            )

    for m_idx, month in enumerate(months):
        expected = BASE_NEW_ACCOUNTS_PER_MONTH * scale * MONTHLY[month.month] * 1.03**m_idx
        for _ in range(int(rng.poisson(expected))):
            account_no += 1
            plan = str(rng.choice(PLANS, p=PLAN_P))
            region = str(rng.choice(REGIONS, p=REGION_P))
            emea_currency = "EUR" if rng.random() < 0.6 else "GBP"
            currency = emea_currency if region == "EMEA" else "USD"
            meta = {
                "currency": currency,
                "channel": str(rng.choice(CHANNELS, p=CHANNEL_P[plan])),
                "region": region,
            }
            mu, sigma = SEATS_LOGMEAN[plan]
            seats = max(1, round(float(np.exp(rng.normal(mu, sigma)))))
            annual = rng.random() < 0.2
            support = plan == "Scale" and rng.random() < 0.2
            promo = plan == "Starter" and rng.random() < 0.1
            anchor = int(rng.integers(1, 29))
            hour = int(rng.integers(6, 20))
            tenure = 0
            current = month
            while current <= months[-1]:
                billed = (not annual) or tenure % 12 == 0
                if billed:
                    when = pd.Timestamp(
                        year=current.year, month=current.month, day=anchor, hour=hour
                    )
                    per_seat = PRICE_BOOK[currency][plan] * (12 if annual else 1)
                    discount = seats * per_seat * (0.15 if annual else 0.0)
                    if promo and tenure == 0:
                        discount = seats * per_seat * 0.5
                    cost = _local(COST_PER_SEAT_USD[plan] * (12 if annual else 1), currency, fx)
                    cycle = "annual" if annual else "monthly"
                    items = [
                        (
                            f"{PLAN_CODE[plan]}-{'YR' if annual else 'MO'}",
                            f"{plan} ({cycle})",
                            "Subscription",
                            seats,
                            per_seat,
                            discount,
                            cost,
                        )
                    ]
                    if support:
                        months_billed = 12 if annual else 1
                        items.append(
                            (
                                "ADD-SUPPORT",
                                "Premium Support",
                                "Add-on",
                                months_billed,
                                _local(SUPPORT_PRICE_USD, currency, fx),
                                0.0,
                                _local(SUPPORT_COST_USD, currency, fx),
                            )
                        )
                    if tenure == 0 and plan in ONBOARDING_PRICE_USD and rng.random() < 0.4:
                        price = _local(ONBOARDING_PRICE_USD[plan], currency, fx)
                        items.append(
                            (
                                "SRV-ONBOARD",
                                "Onboarding Package",
                                "Services",
                                1,
                                price,
                                0.0,
                                round(price * ONBOARDING_COST_RATIO, 2),
                            )
                        )
                    add_invoice(when, account_no, items, meta)
                # lifecycle transition at month end
                hazard = CHURN[plan] * (1.5 if tenure < 3 else 0.6 if tenure >= 12 else 1.0)
                if plan == "Starter" and current == CHURN_SPIKE_MONTH:
                    hazard *= 4.0
                if annual:  # annual plans can only churn at renewal, and are stickier
                    renewal_churn = (1 - (1 - hazard) ** 12) * 0.7
                    churned = tenure % 12 == 11 and rng.random() < renewal_churn
                else:
                    churned = rng.random() < hazard
                if churned:
                    break
                if rng.random() < 0.04:
                    seats = int(np.ceil(seats * rng.uniform(1.1, 1.5)))
                if rng.random() < 0.015 and plan != "Scale":
                    plan = PLANS[PLANS.index(plan) + 1]
                tenure += 1
                current = current + 1

    account_no += 1
    add_invoice(
        ENTERPRISE_DEAL_DAY,
        account_no,
        [("ENT-LIC", "Enterprise License", "Enterprise", 1, ENTERPRISE_DEAL_USD, 0.0, 9600.0)],
        {"currency": "USD", "channel": "outbound_sales", "region": "NA"},
    )

    frame = pd.DataFrame(lines)
    frame = frame.sort_values(["Invoice Date", "Line Ref"], kind="stable").reset_index(drop=True)
    gross = (frame["Seats"].astype(float) * frame["Price per Seat"].astype(float)).to_numpy()

    # messy-but-valid: some dates typed as day-first text instead of Excel dates
    n = len(frame)
    draw = rng.random(n)
    dates = frame["Invoice Date"].astype(object).to_numpy()
    text_dates = draw < 0.08
    dates[text_dates] = [
        d.strftime("%d-%m-%Y") if i % 2 else d.strftime("%d/%m/%Y %H:%M")
        for i, d in enumerate(dates[text_dates])
    ]
    frame["Invoice Date"] = dates
    frame = frame.astype(object)

    anomalies = [
        Anomaly(
            "starter_churn_spike", "drop", "net_revenue", "2026-03-01", "2026-03-31", "WARNING",
            "A Starter price change quadrupled Starter churn in Feb 2026; MRR dips from March.",
            {"category": "Subscription"},
            "month",
        ),
        Anomaly(
            "enterprise_deal", "spike", "net_revenue", "2026-06-15", "2026-06-15", "CRITICAL",
            "One-off USD 48k enterprise license invoice.",
            {"channel": "outbound_sales"},
        ),
    ]  # fmt: skip
    messy = {
        "excel_native_dates": int((~text_dates).sum()),
        "dayfirst_text_dates": int(text_dates.sum()),
        "numeric_account_ids": n,
        "fuzzy_header": 1,
    }
    return DemoTable(
        dataset="saas",
        frame=frame,
        columns=COLUMNS,
        gross=gross.astype(np.float64),
        anomalies=anomalies,
        messy_valid=messy,
    )
