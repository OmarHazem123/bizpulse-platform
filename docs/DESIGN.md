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
| Blank | `""`, whitespace, `nan`, `null`, `none`, `n/a`, `na`, `-` → missing |
| Amount | strip symbols/codes (`$ € £ ¥ E£ USD EUR …`), detect the currency hint. If both `.` and `,` appear, the last one is the decimal separator. If only `,` followed by exactly 1–2 digits, it is a decimal comma, otherwise a thousands separator. `(x)` or trailing `-` means negative. |
| Currency | uppercase ISO code, or a symbol mapped to a code (`$→USD`, `€→EUR`, `£→GBP`, `¥→JPY`, `E£→EGP`). Precedence: currency column, then amount symbol hint, then dataset default. |
| Date | ISO fast path, then `format="mixed"` with `dayfirst`, then Excel serials (1–80 000 days from 1899-12-30). tz-aware is converted to UTC and made naive. |
| ID | trimmed. Excel float artefacts `1001.0` → `1001`. |
| Text dims | trimmed, internal whitespace collapsed. `channel` becomes lower snake case. |
| Status | case/whitespace-insensitive synonym map: paid/complete/completed/fulfilled/closed/success/settled → `completed`; refund/refunded/returned/chargeback/reversed → `refunded`; cancelled/canceled/void/voided/failed/declined/abandoned → `cancelled` |

## 5. Hashing & keys

- `file_sha256`: streamed SHA-256 of the input bytes.
- `transaction_key = sha256("<dataset>|<transaction_id>")`
- `customer_key = sha256("<dataset>|customer|<customer_id>")`, and likewise for products.
- `row_hash = sha256(json.dumps(canonical_values, sort_keys=True))`, with floats rounded
  to 6 dp and dates in ISO format, so the hash survives format changes in the source.

## 6. Quarantine & reports

`QualityIssue(field, code, message, value)`. A quarantined row stores the 1-based data-row
number (`source_row`, excluding the header), all issues and the raw record. The Markdown
report is designed for a non-technical reader: a headline pass rate, a reason table
sorted by count, the column mapping table and a sample of rejected rows.

## 7. Dashboard design (Phase 3)

- **Layout**: wide layout, sidebar for dataset and date-range selection, 5 tabs.
- **Visual language**: one accent color per semantic role (revenue, profit, risk). KPI
  tiles show the value, the delta vs. the previous period and a sparkline.
  CRITICAL/WARNING use consistent red/amber.
- **Tabs**: Executive Overview · Customer & Cohort Intelligence · ML Forecasts & Anomaly
  Alerts · What-If Growth Simulator · Company Data Upload & Quality Audit.
- **Performance**: query results cached per `(dataset, filters, warehouse mtime)`.
  Model training is cached per dataset fingerprint.
- **Empty and error states**: every tab renders a helpful message when there is no data,
  never a stack trace.

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
