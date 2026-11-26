# Design

## 1. Design principles

1. **Contracts over conventions.** The canonical `TransactionRecord` is the only shape
   the warehouse accepts. Every vertical (e-commerce, SaaS, F&B, custom) is a *mapping*
   onto it, never a code branch.
2. **Never drop silently.** A row is loaded, or it is quarantined with exact reasons.
   Counts always reconcile: `read = valid + quarantined` and
   `valid = inserted + updated + unchanged`.
3. **Explainable automation.** Auto-detected mappings, anomaly flags and forecasts always
   say *how* they were derived (method, score, baseline, interval).
4. **Determinism.** Seeds, hashed keys and pinned file metadata make every run
   reproducible byte-for-byte, the generator included.
5. **Small, typed, pure functions** around a thin I/O shell (readers, DuckDB, files),
   so most logic is unit-testable without disk.

## 2. Domain model: one contract, many verticals

| Canonical | E-commerce (CSV) | SaaS (Excel) | Retail F&B (Parquet) |
| --- | --- | --- | --- |
| transaction_id | `Line ID` | `Line Ref` | `ticket_id` + `line_no` (composite) |
| order_id | `Order Number` | `Invoice No` | `ticket_id` |
| order_date | `Order Date` (mixed formats) | `Invoice Date` (Excel dates + day-first text) | `ts` (tz-aware strings) |
| customer_id | `Customer ID` | `Account ID` | `loyalty_id` (`GUEST` = anonymous) |
| product_id | `SKU` | `Plan Code` | `item_code` |
| category | `Product Type` | `Product Family` | `menu_category` |
| quantity | `Qty` | `Seats` | `qty` |
| unit_price | `Unit Price` (`$1,299.00`, `45,90 €`) | `Price per Seat` | `price` |
| channel | `Sales Channel` | `Acquisition Channel` | `order_type` |
| region | `Ship Region` | `Region` | `store` |
| status | `Financial Status` (paid/refunded/voided) | `Invoice Status` (paid/refunded/void/failed) | `status` |
| fee_amount | `Processing Fee` | `Payment Fee` | `platform_fee` (delivery commission) |

The three demo mappings are shipped in `src/bizpulse/resources/mappings/`. A custom
upload starts from auto-detection and can be pinned with a user YAML/JSON mapping.

## 3. Mapping config (`MappingConfig`)

```yaml
dataset: ecommerce                 # tenant key (lower snake case)
display_name: Northwind Commerce (demo)
domain: ecommerce                  # ecommerce | saas | retail_fb | custom
reporting_currency: USD
default_currency: USD              # used when no currency column/symbol
dayfirst: false                    # ambiguous dates: 03/04 = 4 March if true
date_format: null                  # optional strict strftime format
negative_quantity_as_refund: false
anonymous_customer_ids: []         # e.g. [GUEST, WALK-IN]
max_quarantine_ratio: 0.5          # quality gate
columns:                           # canonical -> source header (explicit wins)
  transaction_id: Line ID
composite_keys:                    # canonical -> [source columns] joined with '-'
  transaction_id: [ticket_id, line_no]
status_values:                     # extra synonyms -> completed|refunded|cancelled
  voided: cancelled
read_options:
  sheet_name: null
  delimiter: null
```

### Auto-detection heuristics

1. **Normalize headers**: lowercase, strip accents and non-alphanumerics
   (`"Order Date "` → `orderdate`).
2. **Synonym table** per canonical field (e.g. `order_date` ← orderdate, date,
   createdat, invoicedate, transactiondate, timestamp, ts, …). Exact normalized match =
   confidence 1.0.
3. **Fuzzy**: `difflib.SequenceMatcher` ratio against all synonyms. Best ≥ 0.88 wins.
4. **Content**: if `order_date` is still unmapped, choose the unclaimed column where
   ≥ 90 % of non-blank sampled values parse as dates.
5. **Conflict resolution**: assign greedily by confidence. A source column is claimed
   once. Equal-confidence ties for the same field are marked **ambiguous** and left for
   explicit mapping.

## 4. Normalization rules

| Kind | Rule |
| --- | --- |
| Blank | `""`, whitespace, `nan`, `nat`, `null`, `none`, `n/a`, `#n/a`, `-` → missing. (`NA` is *not* blank: it is a valid region code.) |
| Amount | strip symbols (`US$ C$ A$ R$ E£ $ € £ ¥ ₹`) and **known** ISO-4217 codes (`USD 49.99`), and record the currency hint. Arbitrary 3-letter tokens are not stripped, so `ELE-001` stays unparseable. If both `.` and `,` appear, the last one is the decimal separator. If only `,` followed by exactly 1–2 digits, it is a decimal comma (`45,90`). `1,299` is a thousands separator. `1.234.567` is dotted thousands. `(x)` or trailing `-` means negative. |
| Currency | uppercase ISO code, or a symbol mapped to a code (`$→USD`, `€→EUR`, `£→GBP`, `¥→JPY`, `E£→EGP`). Precedence: currency column, then amount symbol hint, then dataset default. |
| Date | native datetimes, `yyyymmdd`, Excel serials (1–80 000 days from 1899-12-30), then an ISO-8601 fast path, then `format="mixed"` with `dayfirst`. tz-aware values are converted to UTC and made naive. Offset-aware and naive strings are parsed **separately**, because pandas' vectorized ISO parser can leak one element's UTC offset onto later naive elements (regression-tested). |
| ID | trimmed. Excel float artefacts `1001.0` → `1001`. |
| Text dims | trimmed, internal whitespace collapsed. `channel` becomes lower snake case. |
| Status | case/whitespace-insensitive synonym map: paid/complete/completed/fulfilled/closed/success/settled → `completed`; refund/refunded/returned/chargeback/reversed → `refunded`; cancelled/canceled/void/voided/failed/declined/abandoned → `cancelled` |

## 5. Hashing & keys

- `file_sha256`: streamed SHA-256 of the input bytes (batch identity).
- `key64(text)` = the first 8 bytes of `sha256(text)` as a signed BIGINT.
- `transaction_key = key64("<dataset>|<transaction_id>")`
- `customer_key = key64("<dataset>|customer|<customer_id>")`, and likewise for products.
- `row_hash` = full SHA-256 hex of the normalized canonical values in a fixed field
  order, joined by U+001F. Floats are rounded to 6 dp, timestamps are ISO, NULL is U+0000.
  The hash is therefore insensitive to source formatting (`"$1,299.00"` ≡ `1299`).

**Key-size trade-off (approved).** Surrogate and join keys keep 64 of SHA-256's 256 bits.
They are deterministic and fully reproducible like the full hash, but the primary-key
indexes and joins work on 8-byte integers instead of 64-character strings. Measured
result: the warehouse went from 58 MB to 22 MB. By the birthday bound, the probability
of any collision among N keys is ≈ N² / 2⁶⁵:

| Rows per warehouse | Collision probability |
| ---: | ---: |
| 100 k | 2.7 × 10⁻¹⁰ |
| 1 M | 2.7 × 10⁻⁸ |
| 100 M | 2.7 × 10⁻⁴ |

A collision would make the PRIMARY KEY reject the load loudly. It would never merge
rows silently. Change detection (`row_hash`) keeps the full 256-bit SHA-256.

## 6. Quarantine & reports

`QualityIssue(field, code, message, value)`. A quarantined row stores the 1-based data-row
number (`source_row`, excluding the header), all issues and the raw record. The Markdown
report is designed for a non-technical reader: a headline pass rate, a reason table
sorted by count, the column mapping table and a sample of rejected rows.

## 7. Dashboard design

- **Layout.** Wide layout. The sidebar holds the dataset, a date range and the trend
  granularity. There are 5 tabs.
- **Tabs.** Executive Overview · Customer & Cohort Intelligence · ML Forecasts &
  Anomaly Alerts · What-If Growth Simulator · Company Data Upload & Quality Audit.
- **Color is assigned by role, not by taste.** `dashboard/palette.py`:
  - Categorical series use a fixed, CVD-validated slot order: blue, orange, aqua,
    amber. Light and dark mode each use their own validated steps of the same hues;
    validation covered lightness band, chroma, protan/deutan separation and contrast,
    against white and Streamlit's dark surface.
  - Magnitude uses one blue ramp. Its low end recedes toward the surface in each mode.
  - Red and amber are reserved for CRITICAL and WARNING, always paired with an icon and
    a label. The app's accent is therefore set to blue: Streamlit's default red accent
    would make a selected "WARNING" chip look critical.
- **Chart grammar** (`dashboard/charts.py`):
  - **No dual axes.** Pareto is a cumulative-share curve with an 80 % line, not bars
    plus a second axis. What-if revenue and contribution get two charts.
  - Ranked bars use one hue, and the tail is folded into "Other".
  - **Emphasis over rainbow:** LTV shows every cohort in gray and the size-weighted
    average in color. Anomaly timelines draw the metric in gray and the alerts in status
    colors.
  - The forecast draws the 80 / 95 % bands, a dashed projection and the seasonal-naive
    baseline.
  - Every chart has a "Table:" expander, so no value is reachable only by hovering.
- **Deviation from the first sketch.** RFM segments are shown as
  share-of-customers vs. share-of-revenue bars, not a treemap. The bars put the
  decision-relevant comparison on one axis, e.g. in the cafe demo, "At Risk = 23 % of
  customers but 35 % of revenue".
- **KPI tiles.** Each tile shows a value, a sparkline at the selected granularity and
  the change vs. the previous equal-length period. The change is shown only when that
  period lies inside the data. Rates show the change in percentage points.
- **Performance.**
  - Tabs render lazily (`st.tabs(on_change="rerun")`), so only the open tab computes.
  - Loaders are cached on `(warehouse path, warehouse mtime, dataset)`.
  - Measured cold costs per dataset: forecast backtest about 1.3 s, anomaly scoring
    2.5–6 s. Measured tab renders: 0.1–2.5 s for the BI tabs and up to
    4.7 s the first time the ML tab opens.
- **Empty and error states.**
  - With no data, tabs 1–4 show a "load demo data / upload" call to action, with a
    one-click demo loader.
  - Each tab runs inside an error boundary that logs the traceback and shows a short
    message instead of a stack trace.
  - Analyses that need more history (forecast, simulator) explain why they are
    unavailable for small uploads.

## 8. Demo data design (deterministic, seed = 42)

Window: **2025-05-01 → 2026-10-31** (18 months, fixed, never `now()`).

| Dataset | Format | Simulation | Injected anomalies (ground truth in manifest) |
| --- | --- | --- | --- |
| `ecommerce` | CSV | daily Poisson acquisition × trend × weekly/annual seasonality (BFCM, holidays); customer repeat purchases with decaying churn hazard; Zipf product popularity; multi-currency (USD/EUR/GBP) | checkout outage (orders −85 %, 3 days), Apparel refund spike (1 week), social viral spike (1 day) |
| `saas` | Excel | monthly cohorts, per-seat plans, churn hazard by plan and tenure, expansions and upgrades, annual prepay, add-ons | Starter churn spike (pricing change), one-off enterprise deal |
| `retail_fb` | Parquet | 3 stores, hourly tickets, weekday/summer seasonality, loyalty members + anonymous guests, delivery platform commission | store closure (2 days), BOGO cold-drinks promo, POS outage |

Dirty rows (~1–2 %) carry exactly **one** defect each, and are recorded with their
expected reason code in `data/raw/manifests/<dataset>.json`. Messy-but-valid rows
(currency strings, mixed date formats, status casing, padded IDs) must load.
