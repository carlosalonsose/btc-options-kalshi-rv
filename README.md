# Cross-Market Relative Value Research

### BTC Options (Deribit) ↔ Prediction Markets (Kalshi)

> Risk-neutral density extraction from the Deribit option smile, executable
> replication via discrete vertical spreads, and per-bin comparison against
> Kalshi hourly binary contracts (KXBTC).

---

## TL;DR

I built an end-to-end research pipeline that prices Kalshi BTC binary contracts
against an SVI-calibrated risk-neutral distribution from Deribit options.
Across the canonical backtest (71 settled hourly events, 7,775 snapshots and
1.38M bin-snapshot observations from the event-driven collector), the model's
`q_deribit` is **20% better in Brier score** (0.00481 vs 0.00602) and **34%
better in log loss** (0.0184 vs 0.0281) than Kalshi mid-prices (event-clustered
95% bootstrap CIs: 13-28% and 28-41%; the interval excludes zero in both).
**However**,
once executable replication via discrete Deribit strikes is enforced and
Kalshi + Deribit fees and per-event de-correlation are applied, the edge
disappears: net PnL is negative at every threshold except a marginal +$0.06
at a 5% threshold (24 buy / 63 sell trades), and only 14% of observations
have an executable Deribit bracket at all.

The project demonstrates **the gap between theoretical fair value and
executable edge in cross-market relative value**: a model can beat the market
on probabilistic scoring and still have no tradable edge.

![Backtest findings — canonical run (71 events, 1.38M observations)](data/reports/findings.png)

Other views of the same run: `notebooks/01_project_guide.ipynb` (reader
guide), the Streamlit app (`bash analytics.sh`, interactive threshold/spread/
fee sliders), and the Tkinter dashboard (`bash dash.sh`, live snapshot view).

---

## Key findings

All numbers below come from the canonical run (`data/reports/backtest_canonical.txt`):
event-driven dataset, stride 1, 71 settled events, 7,775 snapshots, 1,379,512
bin-snapshot observations, base rate `p_yes = 0.0056`. Coverage: 2026-05-11
16:07 UTC to 2026-05-16 15:51 UTC (about five days).

### 1. The model beats Kalshi mids on probabilistic scoring

| Metric | `q_deribit` (model) | `yes_mid` (Kalshi) |
|---|---|---|
| Brier score | **0.004812** | 0.006023 |
| Log loss | **0.018406** | 0.028052 |

That is a 20% Brier and 34% log-loss improvement. Observations within an event
are highly correlated and there are only 71 YES outcomes (one winning bin per
event), so the uncertainty is computed by resampling whole events
(`src/runner/score_ci.py`, 20,000 resamples, output in `data/reports/score_ci.txt`):

| Metric | Improvement over `yes_mid` | 95% CI | Events where model wins |
|---|---|---|---|
| Brier score | 20.1% | 12.9% to 27.9% | 56 / 71 |
| Log loss | 34.4% | 28.0% to 41.4% | 67 / 71 |

The improvement is statistically distinguishable from zero. Caveat: with a
0.56% YES base rate, ~98% of observations sit in the 0–10% bucket, so both
models score well by predicting near-zero; the magnitude is inflated by the
base rate, not by large-probability skill. Restricted to rows where either
probability exceeds 2% (133,720 rows), Brier is 0.0483 vs 0.0605. The CIs cover
sampling noise within these five days only, not a change of market regime.

### 2. Kalshi mids systematically over-predict realized probabilities

In the 0.4–0.5 prediction bucket, Kalshi mids predict 0.461 but realized
frequency is 0.073 (bias **−0.39**). This is microstructural, not a market
failure: spreads are wide and mids are not executable.

### 3. The executable edge against discrete replication is fragile

PnL against `q_*_repl_disc` (the closest Deribit vertical bracketing the bin),
SELL+BUY YES, one trade per event, Kalshi fee `ceil(0.07*p*(1-p))`, Deribit
fee `$0.015`/contract (a flat approximation of the Deribit schedule,
`min(0.03% of notional, 12.5% of premium)`; see `src/model/fees.py`), net of fees:

| Threshold | n buy | n sell | PnL net |
|---|---|---|---|
| 0.5% | 57 | 69 | **−$1.99** |
| 1.0% | 54 | 68 | **−$1.86** |
| 2.0% | 45 | 67 | **−$1.44** |
| 3.0% | 38 | 67 | **−$1.05** |
| 5.0% | 24 | 63 | **+$0.06** |

Net PnL is negative at every threshold except a marginal break-even at 5%.
Liquidity-filtered slices (spread ≤ 1–2¢, size ≥ 50) show small positive PnL
(+$2.3 to +$2.6 over ~40–60 trades) but these are small-sample slices of a
1.38M-row dataset and should be read as "not yet ruled out", not as edge.

### 4. Few observations can be hedged

Only 14.0% of bin-snapshot observations (193,228 / 1,379,512) have an
executable Deribit bracket with valid bid/ask quotes. The rest are
unhedgeable in practice.

### 5. The structural mismatch

Deribit trades strikes every $500–$1000. Kalshi bins are $100 wide. **Perfect
replication of a Kalshi binary in Deribit is impossible** — the closest
construct is a vertical spread paying a triangular ramp over a 5× wider range.
This introduces tracking error that vanilla payoff comparison ignores.

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

## Method in brief

1. **Horizon bridge.** Kalshi lists hourly binaries, Deribit options expire at
   08:00 UTC. The Kalshi bin settling at `T_K` is priced from the nearest
   Deribit expiry `T_D > T_K`, assuming implied vol is horizon-independent
   (variance linear in time).
2. **Density.** Fit an SVI raw smile to OTM Deribit options. Three curves
   (mid, bid, ask) give a theoretical probability and the cost of going long
   or short the replication.
3. **Replication.** Kalshi bins are $100 wide, Deribit strikes $500-$1000
   apart, so each bin is replicated with a discrete vertical spread.
4. **Backtest.** One trade per event, official Kalshi fee plus a constant
   Deribit fee, with a first-timing and a random control against the
   hindsight "best snapshot" bias.
5. **Maker variant.** Posted quotes with adverse-selection fills and a
   Deribit hedge repriced at `t+1`.

The full write-up of the ten design problems and how each one changed the
approach is in [docs/JOURNEY.md](docs/JOURNEY.md).

---

## Quick start

```bash
# Setup (Python 3.9+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Collect data (runs in background, polls every 60s)
bash start.sh   # stop with: bash stop.sh

# Alternative: event-driven collection (polls often, writes only changes)
bash event_start.sh   # stop with: bash event_stop.sh

# Reproduce the committed reference run on the bundled sample subset
bash backtest.sh --snapshots data/sample_snapshots --stride 1 \
                 --out data/reports/backtest_sample.csv

# After accumulating your own snapshots, run the full backtest
bash backtest.sh --snapshots data/event_snapshots --stride 1
bash backtest.sh --all-trades        # disable per-event aggregation
bash backtest.sh --fees-deribit 0.02 # custom Deribit fee

# Maker / adverse-selection backtest (see docs/JOURNEY.md, Problem 9)
.venv/bin/python -m src.runner.passive_fill                 # canonical run
.venv/bin/python -m src.runner.passive_fill --max-tk-min -1 # no near-settle filter

# Bootstrap CIs for the Brier / log-loss comparison (needs the canonical CSV)
.venv/bin/python -m src.runner.score_ci

# Run tests
python3 -m unittest discover -s tests

# Launch interactive analytics (browser opens at localhost:8501)
bash analytics.sh
```

Both collectors use only public REST endpoints, so no API keys are needed.
The raw event-driven history behind the canonical run is ~1.7 GB (about
340 MB per day) and the backtest output CSV is ~330 MB; both are gitignored. A 1-day subset is committed
under `data/sample_snapshots/` so a third party can reproduce a backtest on
`git clone` without external data. The headline numbers above come from the
**canonical run over the full event-driven dataset** (`data/reports/backtest_canonical.txt`,
committed); the sample subset reproduces the same pipeline on less data.

---

## Project structure

```
.
├── src/
│   ├── snapshotter.py              # 60s polling loop → JSON.gz snapshots
│   ├── event_snapshotter.py        # event-driven collector (writes on quote change)
│   ├── io/load.py                  # in-memory loaders, no I/O over network
│   ├── model/
│   │   ├── black_scholes.py        # bs_call, bs_put, bs_digital, implied_vol
│   │   ├── svi.py                  # SVI raw param + fit + analytical derivs
│   │   ├── intraday_q.py           # VLT-scaled range probability
│   │   ├── exec_price.py           # 3-sided SVI fit (bid/mid/ask)
│   │   ├── replication_discrete.py # discrete vertical-spread pricing
│   │   ├── quality.py              # Deribit expiry quality score
│   │   └── fees.py                 # Kalshi + Deribit fee model
│   ├── signal/edge.py              # per-bin edge annotations
│   ├── runner/
│   │   ├── analyze.py              # pure pipeline + CLI
│   │   ├── backtest.py             # offline PnL evaluation (taker)
│   │   ├── passive_fill.py         # maker / adverse-selection backtest
│   │   └── visualize_findings.py   # static 4-panel plot
│   └── ui/
│       ├── dashboard.py            # Tkinter live view
│       └── streamlit_app.py        # interactive analytics
├── data/
│   ├── sample_snapshots/           # committed 1-day subset (reproducible backtest)
│   └── reports/                    # committed canonical run output + findings.png
├── docs/JOURNEY.md                 # the ten design problems and how each was solved
├── notebooks/01_project_guide.ipynb # reader-facing guide to the project
├── tests/                          # lightweight model sanity tests
├── CHECKPOINT.md                   # plain-language project explanation
├── LICENSE                         # MIT license
├── start.sh / stop.sh              # snapshotter lifecycle
├── event_start.sh / event_stop.sh  # event-driven snapshotter lifecycle
├── backtest.sh                     # re-run backtest
├── analytics.sh                    # launch Streamlit
└── dash.sh                         # launch Tkinter dashboard
```

---

## Tech stack

**Core:** Python 3.9, NumPy, SciPy (`brentq`, `least_squares`), Pandas
**Visualization:** Matplotlib, Plotly, Streamlit, Tkinter
**Testing:** Python `unittest`
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
- Bootstrap CI of SVI fits is implemented as an optional diagnostic, but not
  yet used as a default trade filter.
- Expiry quality scoring is implemented, but production use would require
  stricter threshold calibration.
- Hybrid IV + realized-vol level adjustment for short horizons.

**Microstructure (the real issue)**
- 60s REST snapshots are too slow for live execution. WebSocket order book
  feeds are required.
- Adverse selection: signals that survive long enough for a 60s pipeline to
  detect are likely already taken or are toxic flow.
- Deribit option spreads outside ATM are wide; the executable replication
  set is small (14% of bin-snapshots in the canonical run).

**Economics**
- BRTI vs Deribit Index basis is not modeled.
- Path dependence between T_K (Kalshi settle) and T_D (Deribit expiry) is
  not Δ-hedged in the backtest.
- Tracking error from rectangular-vs-triangular payoff is acknowledged but
  not quantified per-trade.

**Sample size**
- 71 settled events over ~5 days (2026-05-11 to 2026-05-16). Larger than the early runs, but with a
  0.56% YES base rate the *effective* sample for the high-probability region
  (where edge would live) is still small. Snapshots within an event are
  correlated; per-event de-correlation mitigates but does not eliminate this.

---

## What I learned

The headline lesson is not the SVI calibration or the binary-contract pricing
math. It's that **edge in cross-market relative value rarely survives
honest execution modeling**. A naive comparison `q_deribit vs yes_mid` gives
a 20% Brier improvement and looks like alpha; switching to `yes_bid/ask vs
discrete-replication` with fees and one trade per event gives net-negative
PnL at every threshold but a marginal 5% break-even.

The interesting questions in this domain are not "what's the fair value?" but
"what fills can I actually get, when, and at what cost?".

Spreads explain much of it. Across all bins the median Kalshi spread is 1 cent,
but on active bins (mid between 2% and 98%) it is 5 cents at the median, 11
cents at the 75th percentile and 36 cents at the 90th (`data/reports/score_ci.txt`).
Crossing that is expensive relative to the gaps a model built from the outside
of the order book can find.

---

## Disclaimer

Research code for educational purposes only. Nothing here is financial advice
or a trading recommendation.

## Author

Carlos Alonso

## License

MIT
