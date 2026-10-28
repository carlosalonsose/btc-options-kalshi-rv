"""
Event-clustered bootstrap CIs for the probabilistic-scoring comparison.

Reads the per-observation CSV written by `backtest.py` and compares `q_deribit`
against the Kalshi `yes_mid` on Brier score and log loss. Observations inside
an event are strongly correlated (same settlement, adjacent snapshots), so the
bootstrap resamples whole events, not rows.

Also reports the Kalshi spread distribution (all rows and active bins), which
the README quotes.

Usage:
  .venv/bin/python -m src.runner.score_ci
  .venv/bin/python -m src.runner.score_ci --csv data/reports/backtest_canonical.csv \
      --out data/reports/score_ci.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6
COLS = ["event_ticker", "ticker", "q_deribit", "yes_mid", "yes_spread", "outcome"]


def _event_sums(df: pd.DataFrame) -> pd.DataFrame:
    q = df["q_deribit"].clip(EPS, 1 - EPS).values
    m = df["yes_mid"].clip(EPS, 1 - EPS).values
    y = df["outcome"].values
    return pd.DataFrame({
        "ev": df["event_ticker"].values,
        "brier_q": (q - y) ** 2,
        "brier_m": (m - y) ** 2,
        "ll_q": -(y * np.log(q) + (1 - y) * np.log(1 - q)),
        "ll_m": -(y * np.log(m) + (1 - y) * np.log(1 - m)),
        "n": 1,
    }).groupby("ev").sum()


def bootstrap(g: pd.DataFrame, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    a = g[["brier_q", "brier_m", "ll_q", "ll_m", "n"]].values
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        s = a[rng.integers(0, len(a), len(a))].sum(axis=0)
        out.append([(s[1] - s[0]) / s[4], 1 - s[0] / s[1],
                    (s[3] - s[2]) / s[4], 1 - s[2] / s[3]])
    r = np.array(out)
    return {"brier_diff": r[:, 0], "brier_rel": r[:, 1],
            "ll_diff": r[:, 2], "ll_rel": r[:, 3]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Event-clustered bootstrap CIs for Brier / log loss")
    p.add_argument("--csv", type=Path, default=Path("data/reports/backtest_canonical.csv"))
    p.add_argument("--out", type=Path, default=Path("data/reports/score_ci.txt"))
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv, usecols=COLS)
    g = _event_sums(df)
    n_ev = len(g)
    res = bootstrap(g, args.n_boot, args.seed)

    def ci(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return lo, hi

    brier_q = g["brier_q"].sum() / g["n"].sum()
    brier_m = g["brier_m"].sum() / g["n"].sum()
    ll_q = g["ll_q"].sum() / g["n"].sum()
    ll_m = g["ll_m"].sum() / g["n"].sum()

    active = df[(df["yes_mid"] > 0.02) & (df["yes_mid"] < 0.98)]["yes_spread"]
    lines = [
        f"source: {args.csv}",
        f"rows={len(df)} events={n_ev} unique_yes_bins={df.loc[df['outcome'] == 1, 'ticker'].nunique()}",
        f"bootstrap: {args.n_boot} resamples of events, seed={args.seed}, 95% percentile CI",
        "",
        f"brier   model={brier_q:.6f} mid={brier_m:.6f}",
        f"  diff (mid - model)      = {brier_m - brier_q:.6f}  CI [{ci(res['brier_diff'])[0]:.6f}, {ci(res['brier_diff'])[1]:.6f}]",
        f"  relative improvement    = {1 - brier_q / brier_m:.1%}  CI [{ci(res['brier_rel'])[0]:.1%}, {ci(res['brier_rel'])[1]:.1%}]",
        f"logloss model={ll_q:.6f} mid={ll_m:.6f}",
        f"  diff (mid - model)      = {ll_m - ll_q:.6f}  CI [{ci(res['ll_diff'])[0]:.6f}, {ci(res['ll_diff'])[1]:.6f}]",
        f"  relative improvement    = {1 - ll_q / ll_m:.1%}  CI [{ci(res['ll_rel'])[0]:.1%}, {ci(res['ll_rel'])[1]:.1%}]",
        "",
        f"events where model beats mid: brier {(g['brier_m'] > g['brier_q']).sum()}/{n_ev}, "
        f"logloss {(g['ll_m'] > g['ll_q']).sum()}/{n_ev}",
        "",
        "kalshi yes_spread (USD per contract)",
        f"  all rows:   median={df['yes_spread'].median():.2f} p90={df['yes_spread'].quantile(.9):.2f}",
        f"  active bins (2% < yes_mid < 98%, n={len(active)}): median={active.median():.2f} "
        f"mean={active.mean():.3f} p75={active.quantile(.75):.2f} p90={active.quantile(.9):.2f}",
    ]
    text = "\n".join(lines)
    print(text)
    args.out.write_text(text + "\n")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
