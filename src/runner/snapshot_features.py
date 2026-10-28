"""
Per-snapshot model features for the volatility ablation.

For every recorded snapshot this stores what `run_pipeline` computed (forward,
horizons, SVI parameters, ATM implied vol, expiry quality) together with the
Deribit BTC/USD index price. Snapshots where the pipeline fails (no open KXBTC
event, too few valid IVs) are kept with NaN features so the index series stays
dense, which the realised-vol baseline needs.

The output CSV is gitignored (like the other data/reports/*.csv).

Usage:
  .venv/bin/python -m src.runner.snapshot_features
  .venv/bin/python -m src.runner.snapshot_features --snapshots data/sample_snapshots \
      --out data/reports/snapshot_features_sample.csv
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.load import list_snapshots, load_snapshot
from src.model.svi import total_variance
from src.runner.analyze import run_pipeline


def _features(path: Path) -> dict:
    snap = load_snapshot(path)
    row = {
        "snap_ts": pd.Timestamp(snap["snapshot_ts_utc"]).tz_localize("UTC")
        if pd.Timestamp(snap["snapshot_ts_utc"]).tzinfo is None
        else pd.Timestamp(snap["snapshot_ts_utc"]),
        "index_price": snap.get("deribit", {}).get("index_btc_usd", {}).get("index_price", math.nan),
    }
    try:
        res = run_pipeline(snap)
    except Exception:
        return row
    p = res["svi_params"]
    w0 = float(total_variance(np.array([0.0]), p)[0])
    row.update({
        "event_ticker": res["event_ticker"],
        "T_K": res["T_K"],
        "T_D": res["T_D"],
        "F": res["F"],
        "spot": res["spot"],
        "svi_a": p.a, "svi_b": p.b, "svi_rho": p.rho, "svi_m": p.m, "svi_sigma": p.sigma,
        "atm_iv": math.sqrt(max(w0, 1e-12) / res["T_D"]),
        "expiry_quality": res["quality"].score,
    })
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Per-snapshot features for the vol ablation")
    ap.add_argument("--snapshots", type=Path, default=Path("data/event_snapshots"))
    ap.add_argument("--out", type=Path, default=Path("data/reports/snapshot_features.csv"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    paths = list_snapshots(args.snapshots)
    print(f"{len(paths)} snapshots, {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_features, paths, chunksize=50))
    df = pd.DataFrame(rows).sort_values("snap_ts").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out}: {len(df)} rows, {int(df['atm_iv'].notna().sum())} with pipeline features")


if __name__ == "__main__":
    main()
