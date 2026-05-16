"""
Dashboard Tkinter sobre el pipeline run_pipeline().

Dos pestañas:
    Pipeline       — vista detallada por evento (smile, bins, edges).
    Edge tracker   — vista de hallazgos: slice ultra-líquido en vivo
                     + 4 plots tipo findings.png + métricas históricas.

Lanzar:
    .venv/bin/python -m src.ui.dashboard
"""
from __future__ import annotations

import math
import queue
import re
import subprocess
import threading
import time
import tkinter as tk
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.io.load import (
    kalshi_open_events,
    latest_snapshot,
    list_snapshots,
)
from src.model.svi import implied_vol_from_params
from src.runner.analyze import run_pipeline


REFRESH_INTERVAL_MS = 30_000          # auto-refresh
STALE_WARN_SECONDS = 5 * 60           # snap_age > 5 min => banner amarillo
SLICE_SPREAD_MAX = 0.03               # filtro slice ultra-líquido
SLICE_SIZE_MIN = 50
EDGE_ALERT_THRESHOLD = 0.02           # edge_sell_yes_exec > 2% => alerta visual
BACKTEST_CSV = Path("data/reports/backtest.csv")

PIPELINE_TABLE_COLS = [
    ("subtitle",            "Bin",          200, "w"),
    ("yes_bid",             "Yes bid",       60, "e"),
    ("yes_ask",             "Yes ask",       60, "e"),
    ("yes_spread",          "Spread",        60, "e"),
    ("yes_bid_size",        "Bid sz",        55, "e"),
    ("yes_ask_size",        "Ask sz",        55, "e"),
    ("q_deribit",           "Q mid",         70, "e"),
    ("q_buy_exec",          "Q buy",         70, "e"),
    ("q_sell_exec",         "Q sell",        70, "e"),
    ("edge_sell_yes_exec",  "edge sell",     75, "e"),
    ("edge_buy_yes_exec",   "edge buy",      75, "e"),
]

EDGE_TABLE_COLS = [
    ("subtitle",            "Bin",          180, "w"),
    ("yes_bid",             "Yes bid",       60, "e"),
    ("yes_ask",             "Yes ask",       60, "e"),
    ("yes_spread",          "Spread",        60, "e"),
    ("yes_bid_size",        "Bid sz",        55, "e"),
    ("q_buy_exec",          "Q buy",         70, "e"),
    ("edge_sell_yes_exec",  "edge SELL",     80, "e"),
]


# ---------------------------------------------------------------- helpers

def _fmt_age(seconds: float) -> str:
    if seconds < 0:
        return "?"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _fmt_dt(ts: pd.Timestamp) -> str:
    if ts is None or pd.isna(ts):
        return "?"
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


_SUBTITLE_PRICE_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)")


def _short_subtitle(s: str) -> str:
    """Convierte '$80,300 to 80,399.99' en '80300-80400' (compacto, legible)."""
    if not isinstance(s, str):
        return "?"
    nums = _SUBTITLE_PRICE_RE.findall(s)
    parsed = []
    for n in nums:
        try:
            parsed.append(int(round(float(n.replace(",", "")))))
        except ValueError:
            continue
    if len(parsed) >= 2:
        return f"{parsed[0]:,}–{parsed[1]:,}"
    if len(parsed) == 1:
        if "above" in s.lower() or s.lstrip().startswith("Above"):
            return f">{parsed[0]:,}"
        if "below" in s.lower() or s.lstrip().startswith("Below"):
            return f"<{parsed[0]:,}"
        return f"{parsed[0]:,}"
    return s[:22]


def _is_in_slice(row: pd.Series) -> bool:
    s = row.get("yes_spread")
    bs = row.get("yes_bid_size")
    asz = row.get("yes_ask_size")
    if not (isinstance(s, (int, float)) and isinstance(bs, (int, float)) and isinstance(asz, (int, float))):
        return False
    return (s <= SLICE_SPREAD_MAX) and (bs >= SLICE_SIZE_MIN) and (asz >= SLICE_SIZE_MIN)


# ---------------------------------------------------------------- Dashboard

class Dashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cross-Market Arbitrage  |  Kalshi vs Deribit")

        # Fit to screen — ask Tk for the real screen size before setting geometry
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        w = min(1400, sw - 40)
        h = min(860, sh - 60)
        root.geometry(f"{w}x{h}+20+30")
        root.minsize(900, 600)

        self._snap = None
        self._results: dict | None = None
        self._task_queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_layout()
        self._poll_queue()
        # macOS Tkinter bug: window renders blank until it receives a Configure
        # event. Force a 1-pixel resize cycle after mainloop starts to trigger
        # the repaint, then kick off the first data refresh.
        self.root.after(150, self._force_repaint)
        self._schedule_auto_refresh()

    # ------------------------------------------------------------ macOS repaint fix

    def _force_repaint(self) -> None:
        """macOS Tkinter renders a blank window until it gets a resize event.
        Nudge the geometry by 1px then back to force the initial paint."""
        try:
            geo = self.root.geometry()          # e.g. "1300x820+20+30"
            parts = geo.replace("+", "x").split("x")
            w, h = int(parts[0]), int(parts[1])
            self.root.geometry(f"{w}x{h+1}+20+30")
            self.root.update_idletasks()
            self.root.geometry(f"{w}x{h}+20+30")
            self.root.update_idletasks()
        except Exception:
            pass
        # Now lift the window and kick off the first data load
        self.root.lift()
        self.root.focus_force()
        self.refresh()

    # ------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        # === Top bar ===
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        self.snap_label = ttk.Label(top, text="snapshot: -", font=("TkDefaultFont", 10))
        self.snap_label.pack(side="left", padx=(0, 16))

        ttk.Label(top, text="Evento:").pack(side="left")
        self.event_var = tk.StringVar()
        self.event_combo = ttk.Combobox(top, textvariable=self.event_var,
                                        state="readonly", width=30)
        self.event_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh(reuse_snap=True))
        self.event_combo.pack(side="left", padx=4)

        self.refresh_btn = ttk.Button(top, text="Refresh", command=self.refresh)
        self.refresh_btn.pack(side="left", padx=8)

        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto 30s", variable=self.auto_var).pack(side="left")

        self.snapshotter_label = ttk.Label(top, text="snapshotter: ?", foreground="gray")
        self.snapshotter_label.pack(side="right", padx=8)

        # === Stale-data banner (oculto cuando OK) ===
        self.stale_banner = tk.Label(
            self.root, text="", bg="#ffd84a", fg="black",
            font=("TkDefaultFont", 11, "bold"), padx=8, pady=4,
        )
        # solo se hace pack cuando hay alerta

        # === Notebook con 2 pestañas ===
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self.pipeline_frame = ttk.Frame(self.notebook)
        self.edge_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_frame, text="Pipeline")
        self.notebook.add(self.edge_frame, text="Edge tracker")

        self._build_pipeline_tab(self.pipeline_frame)
        self._build_edge_tab(self.edge_frame)

        # === Status bar ===
        self.status = ttk.Label(self.root, text="iniciando pipeline...", anchor="w",
                                relief="sunken", padding=(8, 2))
        self.status.pack(fill="x", side="bottom")

    def _build_pipeline_tab(self, parent: ttk.Frame) -> None:
        info = ttk.Frame(parent, padding=(0, 0, 0, 4))
        info.pack(fill="x")
        for col in range(3):
            info.columnconfigure(col, weight=1)

        self.svi_box = ttk.LabelFrame(info, text="SVI fit  (bid / mid / ask)", padding=6)
        self.svi_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.svi_text = tk.Text(self.svi_box, height=7, width=44, state="disabled",
                                font=("Menlo", 10))
        self.svi_text.pack(fill="both", expand=True)

        self.diag_box = ttk.LabelFrame(info, text="Diagnostico strip", padding=6)
        self.diag_box.grid(row=0, column=1, sticky="nsew", padx=4)
        self.diag_text = tk.Text(self.diag_box, height=7, width=40, state="disabled",
                                 font=("Menlo", 10))
        self.diag_text.pack(fill="both", expand=True)

        self.top_box = ttk.LabelFrame(
            info, text="Top oportunidades  (slice líquido, edge_sell_yes_exec)", padding=6,
        )
        self.top_box.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        self.top_text = tk.Text(self.top_box, height=7, width=46, state="disabled",
                                font=("Menlo", 9))
        self.top_text.pack(fill="both", expand=True)

        # main split: plot left, table right
        main = ttk.Panedwindow(parent, orient="horizontal")
        main.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(main)
        main.add(plot_frame, weight=3)
        self.fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        self.ax_smile = self.fig.add_subplot(2, 1, 1)
        self.ax_edges = self.fig.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        table_frame = ttk.Frame(main)
        main.add(table_frame, weight=2)
        cols = [c[0] for c in PIPELINE_TABLE_COLS]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=20)
        for key, label, width, anchor in PIPELINE_TABLE_COLS:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=anchor)
        vs = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")

    def _build_edge_tab(self, parent: ttk.Frame) -> None:
        # Top: alert + métrica resumen del slice
        summary = ttk.Frame(parent, padding=(0, 0, 0, 4))
        summary.pack(fill="x")

        self.edge_alert = tk.Label(
            summary, text="(sin alertas)", bg="#eeeeee", fg="black",
            font=("TkDefaultFont", 11, "bold"), padx=10, pady=6, anchor="w",
        )
        self.edge_alert.pack(fill="x", padx=2, pady=2)

        metrics_row = ttk.Frame(summary)
        metrics_row.pack(fill="x", pady=(2, 0))
        self.metric_labels: dict[str, ttk.Label] = {}
        for i, (key, title) in enumerate([
            ("n_bins_total",    "bins totales"),
            ("n_bins_slice",    "bins en slice líq."),
            ("n_with_edge",     "bins c/ edge>1%"),
            ("backtest_n",      "trades histórico"),
            ("backtest_pnl",    "PnL histórico"),
            ("backtest_winrate","winrate hist."),
        ]):
            box = ttk.LabelFrame(metrics_row, text=title, padding=(8, 2))
            box.grid(row=0, column=i, sticky="nsew", padx=2)
            metrics_row.columnconfigure(i, weight=1)
            lab = ttk.Label(box, text="-", font=("Menlo", 13, "bold"), anchor="center")
            lab.pack(fill="both")
            self.metric_labels[key] = lab

        # Middle: 4 plots 2x2
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill="both", expand=True, pady=(2, 4))
        self.edge_fig = Figure(figsize=(11, 6.5), dpi=100, tight_layout=True)
        self.ax_e_scatter = self.edge_fig.add_subplot(2, 2, 1)
        self.ax_e_spread = self.edge_fig.add_subplot(2, 2, 2)
        self.ax_e_size = self.edge_fig.add_subplot(2, 2, 3)
        self.ax_e_pnl = self.edge_fig.add_subplot(2, 2, 4)
        self.edge_canvas = FigureCanvasTkAgg(self.edge_fig, master=plot_frame)
        self.edge_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bottom: tabla del slice líquido
        bottom = ttk.LabelFrame(
            parent, text=(
                f"Slice ultra-líquido en vivo  |  spread ≤ {SLICE_SPREAD_MAX*100:.0f}c   "
                f"size ≥ {SLICE_SIZE_MIN}   |   verde = edge SELL > 1%"
            ),
            padding=4,
        )
        bottom.pack(fill="x", pady=(0, 2))
        cols = [c[0] for c in EDGE_TABLE_COLS]
        self.edge_tree = ttk.Treeview(bottom, columns=cols, show="headings", height=8)
        for key, label, width, anchor in EDGE_TABLE_COLS:
            self.edge_tree.heading(key, text=label)
            self.edge_tree.column(key, width=width, anchor=anchor)
        evs = ttk.Scrollbar(bottom, orient="vertical", command=self.edge_tree.yview)
        self.edge_tree.configure(yscrollcommand=evs.set)
        self.edge_tree.pack(side="left", fill="both", expand=True)
        evs.pack(side="right", fill="y")

    # -------------------------------------------------------- background

    def _set_status(self, msg: str) -> None:
        self.status.configure(text=msg)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.refresh_btn.configure(state="disabled" if busy else "normal")

    def refresh(self, reuse_snap: bool = False) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._set_status("recargando...")
        event_ticker = self.event_var.get() or None
        threading.Thread(
            target=self._refresh_worker,
            args=(reuse_snap, event_ticker),
            daemon=True,
        ).start()

    def _refresh_worker(self, reuse_snap: bool, event_ticker: str | None) -> None:
        try:
            t0 = time.monotonic()
            if reuse_snap and self._snap is not None:
                snap = self._snap
            else:
                snap = latest_snapshot()
            results = run_pipeline(snap, event_ticker=event_ticker)
            elapsed = time.monotonic() - t0
            snapshotter_status = self._check_snapshotter()
            self._task_queue.put(("ok", snap, results, elapsed, snapshotter_status))
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] FAILED: {e}\n{tb}", flush=True)
            self._task_queue.put(("err", str(e), tb))

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._task_queue.get_nowait()
                if msg[0] == "ok":
                    _, snap, results, elapsed, snapshotter_status = msg
                    self._snap = snap
                    self._results = results
                    try:
                        self._render(elapsed, snapshotter_status)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[main] RENDER FAILED: {e}\n{tb}", flush=True)
                        self._set_status(f"render error: {e}")
                elif msg[0] == "err":
                    _, err, tb = msg
                    self._set_status(f"error: {err}")
                    print(tb, flush=True)
                    messagebox.showerror("Pipeline error", err)
                self._set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _schedule_auto_refresh(self) -> None:
        if self.auto_var.get() and not self._busy:
            self.refresh()
        self.root.after(REFRESH_INTERVAL_MS, self._schedule_auto_refresh)

    def _check_snapshotter(self) -> str:
        try:
            r = subprocess.run(
                ["pgrep", "-f", "src.snapshotter"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip():
                return f"snapshotter ON (PID {r.stdout.strip().split()[0]})"
            return "snapshotter OFF"
        except Exception:
            return "snapshotter ?"

    # -------------------------------------------------------------- render

    def _render(self, elapsed: float, snapshotter_status: str) -> None:
        if self._results is None or self._snap is None:
            return
        r = self._results

        # --- top bar ---
        snap_age = (datetime.now(timezone.utc) - r["snap_ts"].to_pydatetime()).total_seconds()
        snap_count = len(list_snapshots())
        self.snap_label.configure(
            text=f"snapshot {_fmt_dt(r['snap_ts'])}   age {_fmt_age(snap_age)}   total disk {snap_count}"
        )
        self.snapshotter_label.configure(
            text=snapshotter_status,
            foreground="green" if "ON" in snapshotter_status else "firebrick",
        )

        # --- stale banner ---
        if snap_age > STALE_WARN_SECONDS or "OFF" in snapshotter_status:
            self.stale_banner.configure(
                text=f"⚠ DATOS STALE  —  snap age {_fmt_age(snap_age)}  ·  {snapshotter_status}  ·  no operes",
            )
            if not self.stale_banner.winfo_ismapped():
                self.stale_banner.pack(fill="x", before=self.notebook)
        else:
            if self.stale_banner.winfo_ismapped():
                self.stale_banner.pack_forget()

        # --- event dropdown ---
        events_df = kalshi_open_events(self._snap)
        tickers = events_df["event_ticker"].tolist() if not events_df.empty else []
        if list(self.event_combo["values"]) != tickers:
            self.event_combo.configure(values=tickers)
        if r["event_ticker"] in tickers:
            self.event_var.set(r["event_ticker"])

        # --- per-tab render ---
        self._render_pipeline_tab(r)
        self._render_edge_tab(r)

        # --- status ---
        self._set_status(f"actualizado en {elapsed:.2f}s  |  {datetime.now().strftime('%H:%M:%S')}")

    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    # ---------------------------------------------- pipeline tab render

    def _render_pipeline_tab(self, r: dict) -> None:
        ttl_s = (r["settle"] - datetime.now(timezone.utc)).total_seconds()
        ttl_d = (r["expiry_dt"] - datetime.now(timezone.utc)).total_seconds()
        svi_p = r["svi_params"]
        ts = r.get("two_sided")
        bid_pts = ts.n_points_bid if ts else 0
        ask_pts = ts.n_points_ask if ts else 0
        mid_pts = ts.n_points_mid if ts else r["svi_info"]["n_points"]
        svi_msg = (
            f"event       {r['event_ticker']}\n"
            f"T_K          {_fmt_age(ttl_s)} (settle {r['settle'].strftime('%H:%M:%SZ')})\n"
            f"Deribit T_D  {_fmt_age(ttl_d)} (exp {r['expiry_dt'].strftime('%m-%d %H:%MZ')})\n"
            f"spot/F       {r['spot']:,.2f}  /  {r['F']:,.2f}\n"
            f"SVI mid pts={mid_pts}  bid pts={bid_pts}  ask pts={ask_pts}\n"
            f"a={svi_p.a:+.4f} b={svi_p.b:.4f} rho={svi_p.rho:+.3f}\n"
            f"m={svi_p.m:+.4f} sigma={svi_p.sigma:.4f}"
        )
        self._set_text(self.svi_text, svi_msg)

        s = r["summary"]
        diag_lines = [
            f"n_bins         {s.get('n_bins'):>6}",
            f"sum_q_deribit  {s.get('sum_q_deribit', float('nan')):>6.4f}  (debe ≈ 1)",
            f"sum_yes_mid    {s.get('sum_yes_mid', float('nan')):>6.4f}",
            f"sum_yes_bid    {s.get('sum_yes_bid', float('nan')):>6.4f}",
            f"sum_yes_ask    {s.get('sum_yes_ask', float('nan')):>6.4f}",
            f"|edge_mid| avg {s.get('mean_abs_edge_mid', float('nan')):>6.4f}",
            f"|edge_mid| max {s.get('max_abs_edge_mid', float('nan')):>6.4f}",
        ]
        self._set_text(self.diag_text, "\n".join(diag_lines))

        # Top oportunidades del SLICE LÍQUIDO usando edge_sell_yes_exec
        ann = r["annotated"]
        if "edge_sell_yes_exec" in ann.columns:
            mask_slice = ann.apply(_is_in_slice, axis=1)
            ann_slice = ann[mask_slice]
            top_sell = ann_slice[ann_slice["edge_sell_yes_exec"] > 0.005].copy()
            top_sell = top_sell.sort_values("edge_sell_yes_exec", ascending=False).head(5)
            top_lines: list[str] = []
            if not top_sell.empty:
                top_lines.append("SELL YES + LONG vertical Deribit  (slice líquido):")
                for _, row in top_sell.iterrows():
                    top_lines.append(
                        f"  {_short_subtitle(row.get('subtitle','?')):<14}  "
                        f"yes_bid={row.get('yes_bid',float('nan')):.3f} "
                        f"q_buy={row.get('q_buy_exec',float('nan')):.3f} "
                        f"edge={row['edge_sell_yes_exec']:+.3f}"
                    )
            else:
                top_lines.append("(sin oportunidades en slice líquido en este snapshot)")
                top_lines.append(
                    f"slice = spread ≤ {SLICE_SPREAD_MAX*100:.0f}c  +  size ≥ {SLICE_SIZE_MIN}"
                )
            self._set_text(self.top_text, "\n".join(top_lines))
        else:
            self._set_text(self.top_text, "(reinicia tras Fase 3.5: faltan q_*_exec)")

        self._render_pipeline_plots(r)
        self._render_pipeline_table(ann)

    def _render_pipeline_plots(self, r: dict) -> None:
        self.ax_smile.clear()
        self.ax_edges.clear()

        fit_df = r["fit_df"]
        svi = r["svi_params"]
        T_D = r["T_D"]

        k_grid = np.linspace(fit_df["k"].min() - 0.05, fit_df["k"].max() + 0.05, 300)
        iv_grid = implied_vol_from_params(k_grid, svi, T_D)
        self.ax_smile.scatter(fit_df["k"], fit_df["iv"], alpha=0.7, s=24,
                              label="market IV (mid OTM)", color="tab:blue")
        self.ax_smile.plot(k_grid, iv_grid, color="black", lw=2, label="SVI mid")

        ts = r.get("two_sided")
        if ts is not None and ts.bid is not None:
            iv_b = implied_vol_from_params(k_grid, ts.bid, T_D)
            self.ax_smile.plot(k_grid, iv_b, color="tab:green", lw=1, ls="--", alpha=0.7,
                               label="SVI bid")
        if ts is not None and ts.ask is not None:
            iv_a = implied_vol_from_params(k_grid, ts.ask, T_D)
            self.ax_smile.plot(k_grid, iv_a, color="firebrick", lw=1, ls="--", alpha=0.7,
                               label="SVI ask")

        self.ax_smile.set_xlabel("log-moneyness k = ln(K/F)")
        self.ax_smile.set_ylabel("Implied vol")
        self.ax_smile.set_title(
            f"SVI smile  Deribit {r['expiry_dt'].strftime('%Y-%m-%d %H:%MZ')}  "
            f"(T_D={r['T_D']*365*24:.1f}h)"
        )
        self.ax_smile.grid(True, alpha=0.3)
        self.ax_smile.legend(fontsize=8)

        ann = r["annotated"]
        mids = ((ann["lower"].fillna(ann["upper"]) +
                 ann["upper"].fillna(ann["lower"])) / 2.0)
        relevance = (ann["q_deribit"].fillna(0) > 0.005) | (ann["yes_mid"].fillna(0) > 0.02)
        ann_r = ann[relevance].copy()
        mids_r = mids[relevance]

        width = 80
        self.ax_edges.bar(mids_r - width / 2, ann_r["q_deribit"].values,
                          width=width, color="tab:orange", alpha=0.85, label="Q Deribit")
        if "yes_mid" in ann_r.columns:
            self.ax_edges.bar(mids_r + width / 2, ann_r["yes_mid"].values,
                              width=width, color="tab:blue", alpha=0.6, label="Kalshi mid")
        self.ax_edges.set_xlabel("Bin midpoint (USD)")
        self.ax_edges.set_ylabel("probabilidad")
        self.ax_edges.set_title(
            f"{r['event_ticker']}  -  Q vs Kalshi por bin "
            f"(F={r['F']:,.0f}, T_K={r['T_K']*365*24*60:.1f}min)"
        )
        self.ax_edges.axvline(r["F"], color="gray", linestyle="--", lw=1, alpha=0.7)
        self.ax_edges.legend(fontsize=9)
        self.ax_edges.grid(True, alpha=0.3)

        self.canvas.draw()
        self.canvas.get_tk_widget().update()

    def _render_pipeline_table(self, ann: pd.DataFrame) -> None:
        for it in self.tree.get_children():
            self.tree.delete(it)
        if ann.empty:
            return
        relevance = (ann["q_deribit"].fillna(0) > 0.001) | (ann["yes_mid"].fillna(0) > 0.01)
        df = ann[relevance].copy().sort_values("lower", na_position="first")
        for _, row in df.iterrows():
            values = []
            for key, _label, _w, _a in PIPELINE_TABLE_COLS:
                v = row.get(key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    values.append("")
                elif key == "subtitle":
                    values.append(_short_subtitle(v) if isinstance(v, str) else str(v))
                elif key in {"yes_bid_size", "yes_ask_size"}:
                    values.append(f"{int(v)}" if isinstance(v, (int, float)) else str(v))
                elif isinstance(v, float):
                    if key in {"q_deribit", "q_buy_exec", "q_sell_exec",
                               "yes_bid", "yes_ask", "yes_spread"}:
                        values.append(f"{v:.4f}")
                    else:
                        values.append(f"{v:+.4f}")
                else:
                    values.append(str(v))
            tag = ""
            edge_sell = row.get("edge_sell_yes_exec")
            if isinstance(edge_sell, (int, float)) and not math.isnan(edge_sell):
                if edge_sell > 0.02:
                    tag = "rich"
                elif edge_sell > 0.005:
                    tag = "warm"
            if not tag and _is_in_slice(row):
                tag = "liquid"
            self.tree.insert("", "end", values=values, tags=(tag,))
        self.tree.tag_configure("rich", foreground="white", background="#2e7d32")
        self.tree.tag_configure("warm", foreground="forestgreen")
        self.tree.tag_configure("liquid", background="#fff8e1")

    # ---------------------------------------------- edge tab render

    def _render_edge_tab(self, r: dict) -> None:
        ann = r["annotated"]
        if "q_buy_exec" not in ann.columns:
            for k in self.metric_labels:
                self.metric_labels[k].configure(text="-")
            self.edge_alert.configure(text="(reinicia: faltan q_*_exec)", bg="#eeeeee")
            return

        mask_slice = ann.apply(_is_in_slice, axis=1)
        ann_slice = ann[mask_slice].copy()

        # === metrics ===
        n_total = len(ann)
        n_slice = int(mask_slice.sum())
        with_edge = ann_slice[ann_slice["edge_sell_yes_exec"] > 0.01]
        n_edge = len(with_edge)
        self.metric_labels["n_bins_total"].configure(text=f"{n_total}")
        self.metric_labels["n_bins_slice"].configure(text=f"{n_slice}")
        self.metric_labels["n_with_edge"].configure(
            text=f"{n_edge}",
            foreground="forestgreen" if n_edge > 0 else "black",
        )

        # === alert ===
        big_edge = ann_slice[ann_slice["edge_sell_yes_exec"] > EDGE_ALERT_THRESHOLD]
        if not big_edge.empty:
            top = big_edge.sort_values("edge_sell_yes_exec", ascending=False).iloc[0]
            self.edge_alert.configure(
                text=(f"⚡ EDGE  {_short_subtitle(top.get('subtitle','?'))}   "
                      f"yes_bid={top.get('yes_bid',float('nan')):.3f}   "
                      f"q_buy={top.get('q_buy_exec',float('nan')):.3f}   "
                      f"edge SELL = +{top['edge_sell_yes_exec']:.3f}   "
                      f"size_bid={int(top.get('yes_bid_size',0))}"),
                bg="#a5d6a7",
            )
        else:
            self.edge_alert.configure(
                text=(f"sin edge > {EDGE_ALERT_THRESHOLD*100:.0f}% en slice líquido en este snapshot   "
                      f"({n_slice} bins en slice / {n_total} totales)"),
                bg="#eeeeee",
            )

        # === plots 2x2 ===
        self._render_edge_plots(r, ann, mask_slice)

        # === slice table ===
        self._render_edge_table(ann_slice)

        # === backtest histórico ===
        self._render_backtest_metrics()

    def _render_edge_plots(self, r: dict, ann: pd.DataFrame, mask_slice: pd.Series) -> None:
        for ax in (self.ax_e_scatter, self.ax_e_spread, self.ax_e_size, self.ax_e_pnl):
            ax.clear()

        ann_liq = ann[mask_slice]
        ann_oth = ann[~mask_slice]

        # --- A: scatter q_buy_exec vs yes_bid ---
        ax = self.ax_e_scatter
        if not ann_oth.empty:
            ax.scatter(ann_oth["q_buy_exec"], ann_oth["yes_bid"],
                       s=8, alpha=0.25, color="lightgray", label=f"otros (n={len(ann_oth)})")
        if not ann_liq.empty:
            edge_pos = ann_liq["edge_sell_yes_exec"] > 0.005
            ax.scatter(ann_liq.loc[~edge_pos, "q_buy_exec"], ann_liq.loc[~edge_pos, "yes_bid"],
                       s=55, alpha=0.7, color="tab:gray", edgecolors="black", lw=0.5,
                       label=f"slice (n={len(ann_liq)})")
            if edge_pos.any():
                ax.scatter(ann_liq.loc[edge_pos, "q_buy_exec"],
                           ann_liq.loc[edge_pos, "yes_bid"],
                           s=90, alpha=0.95, color="forestgreen",
                           edgecolors="black", lw=0.7, marker="*",
                           label=f"slice + edge>0.5%  (n={int(edge_pos.sum())})")
        lim = max(0.1, float(ann["yes_bid"].fillna(0).max()) * 1.05,
                  float(ann["q_buy_exec"].fillna(0).max()) * 1.05)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="y = x")
        ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.06, color="green")
        ax.set_xlim(-0.005, lim)
        ax.set_ylim(-0.005, lim)
        ax.set_xlabel("q_buy_exec  (cubrir LONG en Deribit)")
        ax.set_ylabel("yes_bid Kalshi  (precio venta YES)")
        ax.set_title("A) Snapshot actual: edge = puntos en zona verde")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        # --- B: distribución de yes_spread ---
        ax = self.ax_e_spread
        spread = ann["yes_spread"].dropna().clip(0, 0.5)
        if len(spread):
            ax.hist(spread, bins=40, color="tab:gray", alpha=0.85, edgecolor="white")
            ax.axvline(SLICE_SPREAD_MAX, color="firebrick", lw=2, ls="--",
                       label=f"slice ≤ {SLICE_SPREAD_MAX*100:.0f}c")
            pct_below = (ann["yes_spread"] <= SLICE_SPREAD_MAX).mean() * 100
            ax.text(0.6, 0.85, f"{pct_below:.0f}% en slice",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="lightyellow"))
            ax.legend(fontsize=8)
        ax.set_xlabel("yes_spread [USD]")
        ax.set_title("B) Spread Kalshi (snapshot actual)")
        ax.grid(True, alpha=0.3)

        # --- C: distribución de yes_bid_size ---
        ax = self.ax_e_size
        sizes = ann["yes_bid_size"].dropna().clip(0, 500)
        if len(sizes):
            ax.hist(sizes, bins=40, color="tab:gray", alpha=0.85, edgecolor="white")
            ax.axvline(SLICE_SIZE_MIN, color="firebrick", lw=2, ls="--",
                       label=f"slice ≥ {SLICE_SIZE_MIN}")
            pct_zero = (ann["yes_bid_size"] == 0).mean() * 100
            ax.text(0.4, 0.6, f"{pct_zero:.0f}% sin bid",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="lightyellow"))
            ax.legend(fontsize=8)
        ax.set_xlabel("yes_bid_size (cap a 500)")
        ax.set_title("C) Profundidad bid Kalshi (snapshot actual)")
        ax.grid(True, alpha=0.3)

        # --- D: PnL acumulado histórico ---
        self._render_pnl_panel(self.ax_e_pnl)

        self.edge_canvas.draw()
        self.edge_canvas.get_tk_widget().update()

    def _render_pnl_panel(self, ax) -> None:
        if not BACKTEST_CSV.exists():
            ax.text(0.5, 0.5, "(no hay backtest.csv aún)\nrun: python -m src.runner.backtest",
                    ha="center", va="center", fontsize=10)
            ax.set_title("D) Histórico — PnL acumulado del slice (sell-yes)")
            return
        try:
            bt = pd.read_csv(BACKTEST_CSV)
        except Exception as e:
            ax.text(0.5, 0.5, f"(error leyendo backtest.csv: {e})", ha="center", va="center")
            return
        bt["snap_ts"] = pd.to_datetime(bt["snap_ts"])
        bt = bt.dropna(subset=["yes_bid", "yes_spread", "yes_bid_size",
                                "yes_ask_size", "q_buy_exec", "outcome"])
        liq = bt[
            (bt["yes_spread"] <= SLICE_SPREAD_MAX)
            & (bt["yes_bid_size"] >= SLICE_SIZE_MIN)
            & (bt["yes_ask_size"] >= SLICE_SIZE_MIN)
        ]
        sell = liq[(liq["yes_bid"] - liq["q_buy_exec"]) > 0.01].sort_values("snap_ts").copy()
        if sell.empty:
            ax.text(0.5, 0.5, "(slice + edge>1% sin trades en histórico)",
                    ha="center", va="center")
            ax.set_title("D) Histórico — PnL acumulado del slice")
            return
        sell["pnl"] = sell["yes_bid"] - sell["outcome"]
        sell["cum_pnl"] = sell["pnl"].cumsum()
        sell["t_idx"] = np.arange(1, len(sell) + 1)
        ax.plot(sell["t_idx"], sell["cum_pnl"], "o-", color="forestgreen", lw=2, ms=4)
        ax.axhline(0, color="black", lw=0.7)
        final = sell["cum_pnl"].iloc[-1]
        wr = (sell["outcome"] < 0.5).mean() * 100
        ax.text(0.04, 0.92,
                f"n={len(sell)}\nPnL=${final:+.2f}\nwr={wr:.0f}%\nev={sell['event_ticker'].nunique()}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow"))
        ax.set_xlabel("# trade (orden cronológico)")
        ax.set_ylabel("PnL acumulado [USD]")
        ax.set_title("D) Histórico — PnL acum. slice (sell-yes, th=1%)")
        ax.grid(True, alpha=0.3)

    def _render_edge_table(self, ann_slice: pd.DataFrame) -> None:
        for it in self.edge_tree.get_children():
            self.edge_tree.delete(it)
        if ann_slice.empty:
            return
        df = ann_slice.sort_values("edge_sell_yes_exec", ascending=False)
        for _, row in df.iterrows():
            values = []
            for key, _label, _w, _a in EDGE_TABLE_COLS:
                v = row.get(key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    values.append("")
                elif key == "subtitle":
                    values.append(_short_subtitle(v) if isinstance(v, str) else str(v))
                elif key in {"yes_bid_size"}:
                    values.append(f"{int(v)}")
                elif isinstance(v, float):
                    if key in {"yes_bid", "yes_ask", "yes_spread", "q_buy_exec"}:
                        values.append(f"{v:.4f}")
                    else:
                        values.append(f"{v:+.4f}")
                else:
                    values.append(str(v))
            edge = row.get("edge_sell_yes_exec", 0.0) or 0.0
            tag = ""
            if edge > EDGE_ALERT_THRESHOLD:
                tag = "alert"
            elif edge > 0.01:
                tag = "warm"
            elif edge > 0.005:
                tag = "marginal"
            self.edge_tree.insert("", "end", values=values, tags=(tag,))
        self.edge_tree.tag_configure("alert", background="#a5d6a7", foreground="black")
        self.edge_tree.tag_configure("warm", background="#e8f5e9", foreground="forestgreen")
        self.edge_tree.tag_configure("marginal", foreground="#558b2f")

    def _render_backtest_metrics(self) -> None:
        if not BACKTEST_CSV.exists():
            for k in ("backtest_n", "backtest_pnl", "backtest_winrate"):
                self.metric_labels[k].configure(text="(no csv)")
            return
        try:
            bt = pd.read_csv(BACKTEST_CSV)
        except Exception:
            for k in ("backtest_n", "backtest_pnl", "backtest_winrate"):
                self.metric_labels[k].configure(text="(err)")
            return
        bt = bt.dropna(subset=["yes_bid", "yes_spread", "yes_bid_size",
                                "yes_ask_size", "q_buy_exec", "outcome"])
        liq = bt[
            (bt["yes_spread"] <= SLICE_SPREAD_MAX)
            & (bt["yes_bid_size"] >= SLICE_SIZE_MIN)
            & (bt["yes_ask_size"] >= SLICE_SIZE_MIN)
        ]
        sell = liq[(liq["yes_bid"] - liq["q_buy_exec"]) > 0.01]
        n = len(sell)
        if n == 0:
            for k in ("backtest_n", "backtest_pnl", "backtest_winrate"):
                self.metric_labels[k].configure(text="0")
            return
        pnl = (sell["yes_bid"] - sell["outcome"]).sum()
        wr = (sell["outcome"] < 0.5).mean() * 100
        self.metric_labels["backtest_n"].configure(text=f"{n}")
        self.metric_labels["backtest_pnl"].configure(
            text=f"${pnl:+.2f}",
            foreground="forestgreen" if pnl > 0 else "firebrick",
        )
        self.metric_labels["backtest_winrate"].configure(text=f"{wr:.0f}%")


# ---------------------------------------------------------------- main

def main() -> None:
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
