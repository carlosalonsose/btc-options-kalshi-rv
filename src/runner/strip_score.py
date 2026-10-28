"""
Full-strip scoring and volatility ablation.

The bins of one Kalshi event partition the BTC price line, so each snapshot is
a categorical forecast. Two proper scores are computed per snapshot and
averaged: the log score of the winning bin and the ranked probability score
(RPS, bins ordered by strike).

Why a full-strip comparison: scoring bin by bin against `yes_mid` is dominated
by bins with `yes_bid == 0`, where the mid is `ask / 2` and is not a
probability (see docs/JOURNEY.md, Problem 10). Here every method is turned
into a distribution over the strip: negative values are not possible, the
distribution is floored at FLOOR and renormalised, so all methods get the same
treatment.

Methods:
  model      q_deribit from the backtest CSV (SVI + VLT scaling).
  mkt_quoted Kalshi mid on bins with a real bid, 0 elsewhere, normalised.
  mkt_bid    Kalshi bid, normalised.
  flat_atm   lognormal, constant vol = SVI ATM implied vol, forward F, horizon T_K.
  rv_<h>h    lognormal, constant vol = realised vol of the Deribit index over
             the previous <h> hours (1-minute grid), forward F.

Differences are reported as `baseline - model` (positive favours the model)
with event-clustered bootstrap 95% CIs. Nothing is tuned on this data: all
windows are reported, none selected.

Usage:
  .venv/bin/python -m src.runner.snapshot_features      # once, writes features
  .venv/bin/python -m src.runner.strip_score
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

FLOOR = 1e-4
MIN_PER_YEAR = 365.0 * 24.0 * 60.0
RV_WINDOWS_H = (1, 3, 6, 24)
MIN_VALID_RETURNS = 120   # at least 2h of valid 1-minute returns in the window


def _lognormal_bin_prob(lower, upper, F, T, sigma) -> np.ndarray:
    """P(lower <= S <= upper) for a lognormal with forward F, vol sigma, horizon T.
    NaN lower = open lower tail, NaN upper = open upper tail."""
    sd = sigma * np.sqrt(T)

    def surv(K):
        with np.errstate(divide="ignore", invalid="ignore"):
            return norm.cdf((np.log(F / K) - 0.5 * sigma * sigma * T) / sd)

    s_low = np.where(np.isnan(lower), 1.0, surv(lower))
    s_up = np.where(np.isnan(upper), 0.0, surv(upper))
    return np.clip(s_low - s_up, 0.0, 1.0)


def realised_vol(features: pd.DataFrame, hours: int) -> pd.Series:
    """Annualised realised vol of the Deribit index over the previous `hours`,
    evaluated at each snapshot time (past data only). Uses a 1-minute grid; a
    return counts only if both of its grid points have a snapshot in the
    previous 90s, so data gaps are not treated as zero returns. NaN until the
    window holds min(half of it, MIN_VALID_RETURNS) valid returns."""
    s = features.dropna(subset=["index_price"])[["snap_ts", "index_price"]]
    s = s.drop_duplicates("snap_ts", keep="last").sort_values("snap_ts")
    s = s.assign(seen=s["snap_ts"])
    grid = pd.DataFrame({"snap_ts": pd.date_range(s["snap_ts"].min().ceil("min"),
                                                  s["snap_ts"].max().floor("min"), freq="min")})
    g = pd.merge_asof(grid, s, on="snap_ts")
    px = g["index_price"].where((g["snap_ts"] - g["seen"]) <= pd.Timedelta(seconds=90))
    r2 = np.log(px).diff() ** 2
    n = hours * 60
    rate = r2.rolling(n, min_periods=min(n // 2, MIN_VALID_RETURNS)).mean() * MIN_PER_YEAR
    g["vol"] = np.sqrt(rate)
    q = features[["snap_ts"]].reset_index().sort_values("snap_ts")
    m = pd.merge_asof(q, g[["snap_ts", "vol"]], on="snap_ts")
    return m.set_index("index")["vol"].reindex(features.index)


def _normalise(p: pd.Series, sid: pd.Series) -> pd.Series:
    """Normalise to a distribution per snapshot, floor, renormalise. Snapshots
    where the method is undefined (NaN) stay NaN instead of becoming uniform."""
    tot = p.groupby(sid).transform("sum", min_count=1)
    undefined = tot.isna()
    p = (p / tot).where(tot > 0, 0.0).clip(lower=FLOOR)
    p = p / p.groupby(sid).transform("sum")
    return p.mask(undefined)


def build_scores(df: pd.DataFrame, methods: dict[str, pd.Series]) -> pd.DataFrame:
    """Per-snapshot log score and RPS for every method. `df` must be sorted
    by (snapshot, strike) and carry `sid`, `outcome`."""
    sid = df["sid"]
    cum_out = df.groupby("sid", sort=False)["outcome"].cumsum()
    out = df.groupby("sid", sort=False).agg(event=("event_ticker", "first"),
                                            ts=("snap_ts", "first"))
    win = df["outcome"] == 1
    for name, p in methods.items():
        p = _normalise(p, sid)
        out["ls_" + name] = -np.log(p[win].groupby(sid[win]).first())
        cum = p.groupby(sid, sort=False).cumsum()
        out["rps_" + name] = ((cum - cum_out) ** 2).groupby(sid).sum(min_count=1)
    return out.dropna(subset=["ls_model"])


def _boot(diff: pd.Series, events: pd.Series, n_boot: int, seed: int = 0):
    g = pd.DataFrame({"ev": events.values, "d": diff.values, "n": 1}).groupby("ev").sum()
    a = g.values
    rng = np.random.default_rng(seed)
    r = np.empty(n_boot)
    for i in range(n_boot):
        s = a[rng.integers(0, len(a), len(a))].sum(axis=0)
        r[i] = s[0] / s[1]
    return g["d"].sum() / g["n"].sum(), np.percentile(r, [2.5, 97.5])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Full-strip scoring and vol ablation")
    ap.add_argument("--csv", type=Path, default=Path("data/reports/backtest_canonical.csv"))
    ap.add_argument("--features", type=Path, default=Path("data/reports/snapshot_features.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/reports/vol_ablation.txt"))
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args(argv)

    cols = ["snap_ts", "event_ticker", "lower", "upper", "yes_mid", "yes_bid",
            "q_deribit", "outcome", "T_K", "F"]
    d = pd.read_csv(args.csv, usecols=cols)
    d["snap_ts"] = pd.to_datetime(d["snap_ts"], utc=True)
    d["_lo"] = d["lower"].fillna(-np.inf)
    d = d.sort_values(["snap_ts", "event_ticker", "_lo"], kind="stable").reset_index(drop=True)
    d["sid"] = d.groupby(["snap_ts", "event_ticker"], sort=False).ngroup()

    feat = pd.read_csv(args.features)
    feat["snap_ts"] = pd.to_datetime(feat["snap_ts"], utc=True)
    feat = feat.sort_values("snap_ts").reset_index(drop=True)
    for h in RV_WINDOWS_H:
        feat[f"rv_{h}h"] = realised_vol(feat, h)
    atm = feat.drop_duplicates("snap_ts").set_index("snap_ts")[["atm_iv"] + [f"rv_{h}h" for h in RV_WINDOWS_H]]
    d = d.join(atm, on="snap_ts")

    n_all = d["sid"].nunique()

    lo, up = d["lower"].values, d["upper"].values
    methods = {
        "model": d["q_deribit"],
        "mkt_quoted": pd.Series(np.where(d["yes_bid"] > 0, d["yes_mid"], 0.0), index=d.index),
        "mkt_bid": d["yes_bid"],
        "flat_atm": pd.Series(_lognormal_bin_prob(lo, up, d["F"].values, d["T_K"].values,
                                                  d["atm_iv"].values), index=d.index),
    }
    for h in RV_WINDOWS_H:
        methods[f"rv_{h}h"] = pd.Series(
            _lognormal_bin_prob(lo, up, d["F"].values, d["T_K"].values, d[f"rv_{h}h"].values),
            index=d.index)
    S = build_scores(d, methods)
    lines = [
        f"source: {args.csv} + {args.features}",
        f"snapshots={len(S)} events={S['event'].nunique()}  floor={FLOOR}  bootstrap={args.n_boot} resamples of events",
        "scores: lower is better; diff = baseline - model, positive favours the model.",
        "Each baseline is compared with the model on the snapshots where the baseline is defined",
        "(realised-vol windows need enough valid minutes of index data; the collector has gaps).",
        "",
        f"{'baseline':11s} {'snaps':>6s} {'events':>6s} | {'logscore: model':>15s} {'baseline':>9s} {'diff [95% CI]':32s} | {'RPS: model':>10s} {'baseline':>9s} {'diff [95% CI]':32s}",
    ]
    for name in methods:
        if name == "model":
            continue
        sub = S.dropna(subset=["ls_" + name, "rps_" + name])
        d1, c1 = _boot(sub["ls_" + name] - sub["ls_model"], sub["event"], args.n_boot)
        d2, c2 = _boot(sub["rps_" + name] - sub["rps_model"], sub["event"], args.n_boot)
        lines.append(
            f"{name:11s} {len(sub):6d} {sub['event'].nunique():6d} | "
            f"{sub['ls_model'].mean():15.4f} {sub['ls_' + name].mean():9.4f} {d1:+8.4f} [{c1[0]:+.4f}, {c1[1]:+.4f}] | "
            f"{sub['rps_model'].mean():10.4f} {sub['rps_' + name].mean():9.4f} {d2:+8.4f} [{c2[0]:+.4f}, {c2[1]:+.4f}]")

    # stability: first vs second half of the sample by event date
    lines += ["", "stability: first vs second half of events (by snapshot time)"]
    cut = S["ts"].median()
    for label, part in [("first half", S[S["ts"] <= cut]), ("second half", S[S["ts"] > cut])]:
        row = f"  {label:11s} events={part['event'].nunique():3d} " + " ".join(
            f"{n}={part['ls_' + n].mean():.3f}" for n in ["model", "mkt_quoted", "flat_atm", "rv_1h"])
        lines.append(row)
    text = "\n".join(lines)
    print(text)
    args.out.write_text(text + "\n")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
