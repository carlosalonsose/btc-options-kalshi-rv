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
`q_deribit` scores better than Kalshi mid-prices over all rows (Brier 0.00481
vs 0.00602), but that result comes from the 95% of rows where Kalshi's bid is
zero and the "mid" is just `ask/2`, which is not a probability. **On the 5% of
rows with a real two-sided quote, the model is not better than the market**
(Brier 0.0853 vs 0.0833, log loss 0.310 vs 0.296; the log-loss gap excludes zero
in favour of Kalshi under an event-clustered bootstrap). Separately, once the
signal is gated by executable Deribit strikes and Kalshi + Deribit fees and
per-event de-correlation are applied, no trading edge remains: net PnL is negative at every threshold except a marginal +$0.06
at a 5% threshold (24 buy / 63 sell trades), and only 14% of observations
have an executable Deribit bracket at all.

The project demonstrates **the gap between theoretical fair value and
executable edge in cross-market relative value**: a comparison that looks like
alpha over all rows can come from a benchmark that is not a probability, and
disappears when evaluated against executable quotes.

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

### 1. The model beats Kalshi mids only where the mid is not a probability

Over all rows the model scores better:

| Metric | `q_deribit` (model) | `yes_mid` (Kalshi) |
|---|---|---|
| Brier score | 0.004812 | 0.006023 |
| Log loss | 0.018406 | 0.028052 |

But 94.9% of rows (1,309,452) have `yes_bid == 0`. There the mid is `ask/2`
(typically 0.005), averaging 0.0112 against a realised YES frequency of 0.0005,
so it is a poor probability estimate and the model wins almost by construction.
The fair comparison is on rows with a real two-sided quote (`yes_bid > 0`).
Observations within an event are highly correlated and there are only 71 YES
outcomes (one winning bin per event), so uncertainty is computed by resampling
whole events (`src/runner/score_ci.py`, 20,000 resamples, 95% CI, output in
`data/reports/score_ci.txt`). `diff` is mid minus model, so positive favours the model:

| Subset | Rows | Brier model / mid | Brier diff, 95% CI | Log-loss diff, 95% CI | Events model wins (Brier) |
|---|---|---|---|---|---|
| All rows | 1,379,512 | 0.00481 / 0.00602 | +0.00121 [+0.00072, +0.00183] | +0.00965 [+0.00777, +0.01203] | 56 / 71 |
| `yes_bid == 0` | 1,309,452 | 0.00051 / 0.00189 | +0.00138 [+0.00090, +0.00201] | +0.01092 [+0.00902, +0.01332] | 68 / 71 |
| `yes_bid > 0` | 70,060 | 0.08528 / 0.08328 | −0.00200 [−0.00468, +0.00035] | −0.01406 [−0.03145, −0.00052] | 23 / 71 |

On real two-sided quotes the model is slightly worse than the mid: the Brier
interval straddles zero and the log-loss interval excludes it in Kalshi's favour.
The honest summary is that the SVI-based probability does not add information
over a liquid Kalshi mid in this sample, and "beats the market" would be an
overstatement. The intervals cover sampling noise within these five days only,
not a change of market regime.

### 1b. Full-strip scoring: the model is tied with the market, and the smile adds nothing

The bins of an event partition the price line, so each snapshot is a categorical
forecast. `src/runner/strip_score.py` scores every method as a distribution over
the strip (all floored at 1e-4 and renormalised) with the log score of the
winning bin and the ranked probability score (RPS). The market baseline uses the
mid only on bins with a real bid (zero elsewhere), then normalises. `diff` is
baseline minus model, so positive favours the model; 95% CIs by event-clustered
bootstrap (`data/reports/vol_ablation.txt`):

| Baseline | Snapshots / events | Log-score diff | RPS diff |
|---|---|---|---|
| Kalshi mid on quoted bins | 7,775 / 71 | +0.177 [−0.119, +0.598] | −0.006 [−0.205, +0.205] |
| Kalshi bid | 7,775 / 71 | +0.203 [−0.106, +0.634] | +0.014 [−0.190, +0.224] |
| Flat lognormal, SVI ATM vol | 7,775 / 71 | −0.003 [−0.013, +0.007] | −0.000 [−0.001, +0.002] |
| Flat lognormal, realised vol 1h | 5,936 / 25 | +0.040 [−0.107, +0.237] | +0.055 [−0.019, +0.164] |
| Flat lognormal, realised vol 3h | 3,953 / 20 | +0.003 [−0.153, +0.129] | +0.026 [−0.025, +0.073] |
| Flat lognormal, realised vol 6h | 4,391 / 27 | +0.062 [−0.070, +0.167] | +0.062 [−0.003, +0.136] |
| Flat lognormal, realised vol 24h | 7,062 / 67 | +0.145 [+0.051, +0.235] | +0.078 [+0.032, +0.125] |

Reading it:
- **Against the market:** tied. No interval excludes zero.
- **Smile versus flat vol:** a lognormal with constant vol equal to the SVI ATM
  vol scores the same as the full SVI model. The smile shape adds no measurable
  information here; the ATM vol level carries the result.
- **Implied versus realised vol:** realised vol is not better than implied ATM
  vol. The 24h window is clearly worse; the shorter windows are
  indistinguishable but rest on 20-27 events, because the collector has gaps and
  a window needs enough valid 1-minute returns.
- **Stability:** every method scores worse in the second half of the sample than
  in the first, so absolute scores depend on the period; only the differences
  are informative.

### 2. Kalshi mids systematically over-predict realized probabilities

In the 0.4–0.5 prediction bucket, Kalshi mids predict 0.461 but realized
frequency is 0.073 (bias **−0.39**). This is microstructural, not a market
failure: spreads are wide and mids are not executable.

### 3. The executable edge against discrete replication is fragile

The PnL below is the **Kalshi leg only**: a BUY earns `outcome - yes_ask`, a
SELL earns `yes_bid - outcome`, gated by whether the discrete Deribit
replication price signals an edge, minus Kalshi fees and a flat Deribit fee per
trade. The Deribit leg itself is not simulated, so this tests whether the signal
predicts outcomes, not the PnL of a hedged position. PnL against `q_*_repl_disc` (the closest Deribit vertical bracketing the bin),
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

Net PnL is negative at every threshold except a marginal break-even at 5%
(+$0.06 on 87 one-contract trades, which is noise).
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

The full write-up of the twelve design problems and how each one changed the
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

# Full-strip scoring and vol ablation (features first, then scoring)
.venv/bin/python -m src.runner.snapshot_features
.venv/bin/python -m src.runner.strip_score

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
│   │   ├── score_ci.py             # event-clustered bootstrap CIs, split by quote type
│   │   ├── snapshot_features.py    # per-snapshot F, SVI params, ATM vol, index price
│   │   ├── strip_score.py          # full-strip log score / RPS and vol ablation
│   │   └── visualize_findings.py   # static 4-panel plot
│   └── ui/
│       ├── dashboard.py            # Tkinter live view
│       └── streamlit_app.py        # interactive analytics
├── data/
│   ├── sample_snapshots/           # committed 1-day subset (reproducible backtest)
│   └── reports/                    # committed canonical run output + findings.png
├── docs/JOURNEY.md                 # the twelve design problems and how each was solved
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

**Evaluation**
- Probabilistic scoring against `yes_mid` is only meaningful on rows with a
  two-sided quote. A fairer baseline for zero-bid bins (for example a
  calibrated function of the ask) is not implemented.
- The PnL covers the Kalshi leg only; the Deribit hedge leg and its tracking
  error are not simulated in the taker backtest.
- The SVI fit has no butterfly or calendar no-arbitrage constraints, and the
  digital price uses `N(d2)` with the strike's implied vol, without the
  smile-skew correction. Finding 1b suggests this does not matter in this
  sample, since a flat-vol lognormal scores the same.
- The realised-vol baselines are weak: the collector has gaps, so short windows
  rest on 20-27 events. No basis (BRTI vs Deribit index) model has been tested yet.

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
honest execution modeling**. A naive comparison `q_deribit vs yes_mid` over all rows gives
a 20% Brier improvement and looks like alpha; but that gain comes from zero-bid
rows where the mid is not a probability, and on real two-sided quotes the model
is no better than the market. Switching to `yes_bid/ask vs
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
