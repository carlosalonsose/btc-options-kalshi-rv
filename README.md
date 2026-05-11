# Cross-Market Relative Value Research

### BTC Options (Deribit) ↔ Prediction Markets (Kalshi)

> Risk-neutral density extraction from the Deribit option smile, executable
> replication via discrete vertical spreads, and per-bin comparison against
> Kalshi hourly binary contracts (KXBTC).

---

## TL;DR

I built an end-to-end research pipeline that prices Kalshi BTC binary contracts
against an SVI-calibrated risk-neutral distribution from Deribit options.
Across 575 1-minute snapshots covering 21 settled events (~25h of intraday
data, 18K bin-snapshot observations), the model's `q_deribit` is **46% better
calibrated** (Brier score 0.0042 vs 0.0077) than Kalshi mid-prices. **However**,
once executable replication via discrete Deribit strikes is enforced and
fees + per-event de-correlation are applied, the apparent edge largely
disappears: realistic execution timing yields PnL ~−$0.32 over 12 events,
with one cherry-picking aggregation mode showing +$1.46 (which I show
to be optimistic).

The project demonstrates **the gap between theoretical fair value and
executable edge in cross-market relative value** — a trap most cross-market
arbitrage projects fall into and rarely surface.

---

## Motivation

Two markets price BTC outcomes with different microstructures:

- **Kalshi** offers hourly binary contracts (`KXBTC-YYMMMDDHH-...`) paying $1
  if BTC settles in a $100 range at the top of each hour. ~200 bins per event.
- **Deribit** trades vanilla BTC options (calls/puts at discrete strikes,
  daily/weekly/monthly expiries). The option smile implies a continuous
  risk-neutral distribution.

If both markets are mispriced relative to each other (after accounting for
costs, basis between settlement indices, and execution constraints), there's
exploitable cross-market signal. This project quantifies whether that signal
exists for a small/retail trader operating with REST API snapshots.

---

## Methodology

### Pipeline (`src/runner/analyze.py`)

For each Kalshi event chosen at snapshot time `t`:

1. Identify the next Kalshi settlement time `T_K` (~hourly).
2. Pick the earliest Deribit expiry `T_D > T_K` (typically next-day 08:00 UTC).
3. Compute the put-call-parity forward `F = K + e^{rT}(C - P)` from ATM pairs.
4. Build the OTM-only smile (puts for K<F, calls for K≥F) → invert to IV via
   Brent's method on Black-Scholes.
5. **Calibrate three SVI fits** (Gatheral raw form, 5 params, weighted least
   squares):
   - `svi_mid` from option mid-prices (theoretical baseline).
   - `svi_bid` from bid-side prices (worst case for short Deribit hedge).
   - `svi_ask` from ask-side prices (worst case for long Deribit hedge).
6. For each Kalshi bin `[L, U]`, under VLT scaling (variance linear in time)
   compute three Q-probabilities:
   - `q_deribit = N(d2(L)) - N(d2(U))` using `svi_mid` (theoretical).
   - `q_buy_exec = max(q_under_bid, q_under_ask)` (proxy of "cost to go long").
   - `q_sell_exec = min(...)` (proxy of "cost to go short").
7. **Discrete replication** (`src/model/replication_discrete.py`): find the
   Deribit strikes `K_low ≤ L < U ≤ K_high` that bracket the bin, build the
   actual call vertical, and compute the executable cost normalized to the
   bin's nominal. This is the most honest "fair value executable in Deribit".

### Backtesting (`src/runner/backtest.py`)

For each snapshot in `data/snapshots/`:
1. Run the full pipeline.
2. Join each bin with its actual outcome from Kalshi's `/markets?status=settled`
   endpoint.
3. Compute Brier score, log loss, calibration buckets, reliability by horizon.
4. Simulate executable PnL with:
   - **Fees**: Kalshi official formula `ceil(0.07·p·(1−p))` + Deribit constant.
   - **Per-event aggregation modes**: best-score (optimistic), first-opportunity
     (realistic), random (control). Decorrelates snapshots within an event.
   - **Liquidity grid**: spread × depth filters.

### Visualization

- **`findings.png`** — static 4-panel summary of the spread/liquidity
  distributions and PnL realization.
- **Streamlit app** (`bash analytics.sh`) — interactive sliders for threshold,
  spread, depth, fees, and aggregation mode, with live PnL recompute.
- **Tkinter dashboard** (`bash dash.sh`) — live snapshot view (kept for
  potential paper trading).

---

## Key findings

### 1. The model is well-calibrated

| Metric | `q_deribit` (model) | `yes_mid` (Kalshi) |
|---|---|---|
| Brier score | **0.00417** | 0.00771 |
| Log loss | **0.01418** | 0.03505 |

Reliability diagram: `q_deribit` is nearly perfectly calibrated up to bucket
0.4 (e.g., predicted 0.149 vs realized 0.153). Beyond that, sample size is
the limiting factor (~23 rows total in higher buckets across 21 events).

### 2. Kalshi mids systematically over-predict realized probabilities

In the 0.4–0.5 prediction bucket, Kalshi mids predict 0.450 but realized
frequency is 0.062 (bias **−0.39**). This is microstructural, not a market
failure: spreads are wide and mids are not executable.

### 3. The executable edge against discrete replication is fragile

When PnL is computed against `q_buy_repl_disc` (the cost of the closest
Deribit vertical bracketing the bin):

| Aggregation mode | n trades | PnL net |
|---|---|---|
| Best score per event (optimistic) | 12 | **+$1.46** |
| First opportunity per event (realistic) | 12 | **−$0.32** |
| Random pick per event | 12 | −$1.39 |

**The "edge" of the optimistic mode is largely a cherry-picking artifact** of
hindsight aggregation. With realistic execution timing, the system is
PnL-neutral.

### 4. Only 11.4% of bin-snapshot observations have an executable Deribit
bracket with valid bid/ask quotes. The rest are unhedgeable in practice.

### 5. The structural mismatch

Deribit trades strikes every $500–$1000. Kalshi bins are $100 wide. **Perfect
replication of a Kalshi binary in Deribit is impossible** — the closest
construct is a vertical spread paying a triangular ramp over a 5× wider range.
This introduces tracking error that vanilla payoff comparison ignores.

---

## Project structure

```
.
├── src/
│   ├── snapshotter.py              # 60s polling loop → JSON.gz snapshots
│   ├── io/load.py                  # in-memory loaders, no I/O over network
│   ├── model/
│   │   ├── black_scholes.py        # bs_call, bs_put, bs_digital, implied_vol
│   │   ├── svi.py                  # SVI raw param + fit + analytical derivs
│   │   ├── intraday_q.py           # VLT-scaled range probability
│   │   ├── exec_price.py           # 3-sided SVI fit (bid/mid/ask)
│   │   ├── replication_discrete.py # discrete vertical-spread pricing
│   │   └── fees.py                 # Kalshi + Deribit fee model
│   ├── signal/edge.py              # per-bin edge annotations
│   ├── runner/
│   │   ├── analyze.py              # pure pipeline + CLI
│   │   ├── backtest.py             # offline PnL evaluation
│   │   └── visualize_findings.py   # static 4-panel plot
│   └── ui/
│       ├── dashboard.py            # Tkinter live view
│       └── streamlit_app.py        # interactive analytics
├── data/                           # (gitignored) snapshots, settled cache, reports
├── HANDOVER.md                     # technical handover document
├── CHECKPOINT.md                   # plain-language project explanation
├── start.sh / stop.sh              # snapshotter lifecycle
├── backtest.sh                     # re-run backtest
├── analytics.sh                    # launch Streamlit
└── dash.sh                         # launch Tkinter dashboard
```

---

## Quick start

```bash
# Setup (Python 3.9+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Collect data (runs in background, polls every 60s)
bash start.sh

# After accumulating snapshots, run backtest
bash backtest.sh                     # default: stride=5, max 1 trade/event
bash backtest.sh --all-trades        # disable per-event aggregation
bash backtest.sh --fees-deribit 0.02 # custom Deribit fee

# Launch interactive analytics (browser opens at localhost:8501)
bash analytics.sh
```

---

## Tech stack

**Core:** Python 3.9, NumPy, SciPy (`brentq`, `least_squares`), Pandas
**Visualization:** Matplotlib, Plotly, Streamlit, Tkinter
**APIs:** Kalshi REST (`/trade-api/v2/markets`, `/events`),
Deribit REST (`/public/get_instruments`, `/get_book_summary_by_currency`,
`/get_index_price`)
**Data:** gzipped JSON snapshots (lossless, schema-flexible, no Parquet
dependency)

---

## Limitations and future work

The project is a research artifact, not a production trading system. Known
gaps before any live use:

**Modeling**
- VLT scaling is assumed; SSVI with no-calendar-arbitrage constraints would be
  more rigorous for multi-expiry.
- Bootstrap CI of SVI fits to filter trades where the model is uncertain
  (planned, see HANDOVER.md §10).
- Expiry quality scoring to reject contaminated near-expiry surfaces.
- Hybrid IV + realized-vol level adjustment for short horizons.

**Microstructure (the real issue)**
- 60s REST snapshots are too slow for live execution. WebSocket order book
  feeds are required.
- Adverse selection: signals that survive long enough for a 60s pipeline to
  detect are likely already taken or are toxic flow.
- Deribit option spreads outside ATM are wide; the executable replication
  set is small (~11% of bins).

**Economics**
- BRTI vs Deribit Index basis is not modeled.
- Path dependence between T_K (Kalshi settle) and T_D (Deribit expiry) is
  not Δ-hedged in the backtest.
- Tracking error from rectangular-vs-triangular payoff is acknowledged but
  not quantified per-trade.

**Sample size**
- 21 events / ~25h of intraday data is insufficient for statistical
  significance. The snapshotter is left running for accumulation.

---

## What I learned

The headline lesson is not the SVI calibration or the binary-contract pricing
math. It's that **edge in cross-market relative value rarely survives
honest execution modeling**. A naive comparison `q_deribit vs yes_mid` gives
a 46% Brier improvement and looks like alpha; a comparison `yes_bid vs
discrete-replication ask` with first-opportunity timing and fees gives no
edge at all. The interesting questions in this domain are not "what's the
fair value?" but "what fills can I actually get, when, and at what cost?".

I also internalized why classical arbitrage between fundamentally similar
contracts is rare in retail-accessible markets: the spreads charged by the
venue (Kalshi: 10–30 cents on a $1 contract) are an order of magnitude
larger than typical mispricings detectable from outside the order book.

---

## Author

Carlos Alonso — research project for quant internship / MFE application
portfolio.

## License

MIT
