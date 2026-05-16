"""
Streamlit analytics app.

4 pestañas:
  1. Overview: KPIs + el plot 2x2 de hallazgos en formato Plotly interactivo.
  2. Backtest interactivo: sliders threshold / max_spread / min_size, recompute live.
  3. Live snapshot: snapshot mas reciente + oportunidades del slice liquido AHORA.
  4. Calibracion / sanity: reliability diagram + Brier por horizonte + diag table.

Lanzar:
    bash analytics.sh
    # o
    .venv/bin/streamlit run src/ui/streamlit_app.py
"""
from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

# Asegura que el repo root esta en sys.path independientemente de desde donde
# se lance streamlit (streamlit cd's al dir del script y pierde 'src').
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.io.load import latest_snapshot, list_snapshots
from src.model.fees import kalshi_fee
from src.runner.analyze import run_pipeline


BACKTEST_CSV        = Path("data/reports/backtest.csv")
BACKTEST_EVENT_CSV  = Path("data/reports/backtest_event.csv")
DIAG_CSV            = Path("data/reports/backtest_diag.csv")
DIAG_EVENT_CSV      = Path("data/reports/backtest_event_diag.csv")
SNAPSHOTS_DIR       = Path("data/snapshots")
EVENT_SNAPSHOTS_DIR = Path("data/event_snapshots")

# Mapas fuente → archivos y directorios
_SOURCE_META = {
    "original": {
        "label":   "Original 60s (data/snapshots)",
        "csv":     BACKTEST_CSV,
        "diag":    DIAG_CSV,
        "snap_dir": SNAPSHOTS_DIR,
        "args":    ["--stride", "5"],
        "gen_cmd": "bash backtest.sh --stride 5",
    },
    "event": {
        "label":   "Event-driven (data/event_snapshots)",
        "csv":     BACKTEST_EVENT_CSV,
        "diag":    DIAG_EVENT_CSV,
        "snap_dir": EVENT_SNAPSHOTS_DIR,
        "args":    ["--snapshots", "data/event_snapshots", "--stride", "1",
                    "--out", "data/reports/backtest_event.csv"],
        "gen_cmd": "bash backtest.sh --snapshots data/event_snapshots --stride 1 --out data/reports/backtest_event.csv",
    },
}


# --- helpers ----------------------------------------------------------------

def csv_has_data(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1


@st.cache_data(show_spinner=False)
def load_backtest(csv_path: str, diag_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga CSV de backtest. Parametrizado por ruta para que st.cache_data
    distinga entre fuentes."""
    p = Path(csv_path)
    if not csv_has_data(p):
        return pd.DataFrame(), pd.DataFrame()

    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), pd.DataFrame()

    if "snap_ts" in df.columns:
        df["snap_ts"] = pd.to_datetime(df["snap_ts"], utc=True, errors="coerce")

    dp = Path(diag_path)
    try:
        diag = pd.read_csv(dp) if csv_has_data(dp) else pd.DataFrame()
    except pd.errors.EmptyDataError:
        diag = pd.DataFrame()

    if not diag.empty and "snap_ts" in diag.columns:
        diag["snap_ts"] = pd.to_datetime(diag["snap_ts"], utc=True, errors="coerce")
    return df, diag


def run_backtest_from_app(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "src.runner.backtest", *args]
    return subprocess.run(
        cmd,
        cwd=_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@st.cache_data(show_spinner=False, ttl=30)
def load_latest_snapshot_pipeline() -> dict | None:
    try:
        snap = latest_snapshot()
        return run_pipeline(snap)
    except Exception as e:
        return {"error": str(e)}


def filter_liquid(df: pd.DataFrame, max_spread: float, min_size: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        (df["yes_spread"].notna())
        & (df["yes_bid_size"].notna())
        & (df["yes_ask_size"].notna())
        & (df["yes_spread"] <= max_spread)
        & (df["yes_bid_size"] >= min_size)
        & (df["yes_ask_size"] >= min_size)
    ].copy()


def compute_pnl_rows(df: pd.DataFrame, threshold: float,
                     fee_deribit: float = 0.0,
                     one_per_event: bool = False,
                     event_pick: str = "best",
                     model: str = "exec") -> dict:
    """Sell-yes only PnL aplicando fees Kalshi (formula oficial) + fee Deribit
    constante por contrato.

    model:
        "exec" -> usa q_buy_exec (SVI bid/ask teorica, Sprint 1).
        "repl" -> usa q_buy_repl_disc (replicacion REAL con strikes Deribit,
                  Sprint 1.5; mas conservadora pero mas honesta).

    one_per_event: si True, cada evento aporta a lo sumo un trade.
    event_pick: como elegir el trade dentro del evento:
        "best"   -> el de mejor score (CHERRY-PICK retrospectivo, optimista).
        "first"  -> el primero cronologicamente (REALISTA: vendes al ver edge).
        "random" -> aleatorio (intermedio).
    """
    q_col = "q_buy_repl_disc" if model == "repl" else "q_buy_exec"
    if df.empty:
        return {"n_trades": 0, "pnl_total": 0.0, "trades": pd.DataFrame()}
    d = df.dropna(subset=["yes_bid", q_col, "outcome"]).copy()
    if d.empty:
        return {"n_trades": 0, "pnl_total": 0.0, "trades": pd.DataFrame()}
    d["sell_edge"] = d["yes_bid"] - d[q_col]
    spread = (d["yes_ask"] - d["yes_bid"]).fillna(0)
    d["sell_score"] = d["sell_edge"] - 0.5 * spread
    sell_mask = d["sell_edge"] > threshold
    trades = d[sell_mask].copy()
    if one_per_event and not trades.empty:
        if event_pick == "best":
            trades = (trades.sort_values("sell_score", ascending=False)
                            .drop_duplicates("event_ticker"))
        elif event_pick == "first":
            trades = (trades.sort_values("snap_ts")
                            .drop_duplicates("event_ticker"))
        elif event_pick == "random":
            trades = (trades.sample(frac=1, random_state=42)
                            .drop_duplicates("event_ticker"))
    trades = trades.sort_values("snap_ts")
    if trades.empty:
        return {"n_trades": 0, "pnl_total": 0.0, "winrate": 0.0, "trades": trades}
    trades["fee_kalshi"] = trades["yes_bid"].apply(kalshi_fee)
    trades["fee_total"] = trades["fee_kalshi"] + fee_deribit
    trades["pnl_gross"] = trades["yes_bid"] - trades["outcome"]
    trades["pnl"] = trades["pnl_gross"] - trades["fee_total"]
    trades["cum_pnl"] = trades["pnl"].cumsum()
    return {
        "n_trades": int(len(trades)),
        "pnl_total": float(trades["pnl"].sum()),
        "pnl_gross_total": float(trades["pnl_gross"].sum()),
        "fees_total": float(trades["fee_total"].sum()),
        "avg_pnl": float(trades["pnl"].mean()),
        "winrate": float((trades["outcome"] < 0.5).mean()),
        "n_events": int(trades["event_ticker"].nunique()),
        "trades": trades,
    }


def fmt_age(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"


def _chart(fig) -> None:
    """Wrapper de st.plotly_chart sin kwargs deprecados (Streamlit 1.50).
    Aplica autosize + margenes razonables para que el grafico use el ancho del
    contenedor (columna o pestana) automaticamente."""
    fig.update_layout(autosize=True, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig)


# --- tabs -------------------------------------------------------------------

def render_overview(df: pd.DataFrame, max_spread: float, min_size: float,
                    threshold: float, fee_deribit: float, one_per_event: bool,
                    fee_multiplier: float = 1.0,
                    event_pick: str = "first",
                    model: str = "repl",
                    snap_dir: Path = SNAPSHOTS_DIR) -> None:
    st.subheader("Resumen del estado del proyecto")

    if df.empty:
        st.warning("No hay backtest CSV. Corre `bash backtest.sh` primero.")
        return

    n_snap_disk = len(list_snapshots(snap_dir))
    liq = filter_liquid(df, max_spread, min_size)
    pnl = compute_pnl_rows(liq, threshold, fee_deribit=fee_deribit,
                           one_per_event=one_per_event,
                           event_pick=event_pick, model=model)

    model_caption = {
        "exec": "Modelo: SVI teórico bid/ask (todos los bins, optimista).",
        "repl": "Modelo: replicación REAL con strikes Deribit cotizados (solo ~11% de bins).",
    }[model]
    pick_caption = {
        "first": "Pick: first opportunity (REALISTA — vendes al ver edge).",
        "best": "Pick: best score (OPTIMISTA — cherry-pick retrospectivo).",
        "random": "Pick: random (intermedio).",
    }[event_pick] if one_per_event else "Pick: ALL trades (correlacionados, n inflado)."
    st.caption(f"{model_caption}  ·  {pick_caption}")

    if fee_multiplier != 1.0:
        st.warning(f"Stress test activo: fees ×{fee_multiplier:.1f}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Snapshots disco", f"{n_snap_disk:,}")
    c2.metric("Filas backtest", f"{len(df):,}")
    c3.metric("Eventos cubiertos", f"{df['event_ticker'].nunique()}")
    c4.metric("Trades", f"{pnl['n_trades']}",
              help=f"Threshold {threshold:.1%}, "
                   f"{'1/event' if one_per_event else 'all'}")
    c5.metric("PnL bruto", f"${pnl.get('pnl_gross_total', 0):+.2f}")
    c6.metric("PnL NETO (post fees)", f"${pnl['pnl_total']:+.2f}",
              delta=f"{pnl.get('avg_pnl', 0)*100:+.2f}c/trade"
                    if pnl["n_trades"] > 0 else None)

    st.divider()
    st.subheader("Los 4 paneles del análisis")
    st.markdown(
        "Cada panel responde a una pregunta. Bajo cada gráfico tienes una "
        "explicación de **qué estás mirando** y **cómo interpretarlo**."
    )

    col1, col2 = st.columns(2)

    # ============================== A ==============================
    with col1:
        fig = px.histogram(df, x="yes_spread", nbins=60, range_x=[0, 0.5],
                           title="A) ¿Qué tan estrecho es el spread Kalshi?")
        fig.add_vline(x=max_spread, line_dash="dash", line_color="firebrick",
                      annotation_text=f"tu cutoff = {max_spread:.2f}")
        fig.update_layout(height=350, bargap=0.05,
                          xaxis_title="spread (yes_ask − yes_bid)  [USD]",
                          yaxis_title="# bins")
        _chart(fig)
        pct_below = (df["yes_spread"] <= max_spread).mean() * 100
        st.markdown(
            f"""
            **Qué ves:** cada barra cuenta cuántos bins Kalshi tienen ese spread.
            La línea roja es tu cutoff actual (mueve el slider `max_spread`).

            **Cómo interpretar:** un spread de 10c significa que para cruzarlo
            (ej. comprar al ask, vender al bid) pierdes 10c por contrato de
            entrada. Si el spread es enorme, el "edge" del modelo es ruido frente
            al coste de cruzar.

            **Diagnóstico ahora:** sólo el **{pct_below:.0f}%** de bins
            cumplen `spread ≤ {max_spread:.2f}`. El resto son intradeables.
            """
        )

    # ============================== B ==============================
    with col2:
        sizes = df["yes_bid_size"].clip(0, 500)
        fig = px.histogram(sizes, nbins=50,
                           title="B) ¿Cuántos contratos hay esperando a comprar?")
        fig.add_vline(x=min_size, line_dash="dash", line_color="firebrick",
                      annotation_text=f"tu cutoff = {min_size}")
        fig.update_layout(height=350, bargap=0.05, showlegend=False,
                          xaxis_title="yes_bid_size  (cortado en 500)",
                          yaxis_title="# bins")
        _chart(fig)
        pct_zero = (df["yes_bid_size"] == 0).mean() * 100
        pct_above = (df["yes_bid_size"] >= min_size).mean() * 100
        st.markdown(
            f"""
            **Qué ves:** distribución de la profundidad del bid en Kalshi.
            La barra gigante en 0 = bins sin nadie esperando comprar.

            **Cómo interpretar:** sin bid, **no puedes vender YES** aunque tu
            modelo grite "vende". El bid es la contraparte que necesitas.

            **Diagnóstico ahora:** **{pct_zero:.0f}%** de bins no tienen bid.
            Sólo **{pct_above:.0f}%** tienen `bid_size ≥ {min_size}`.
            """
        )

    # ============================== C ==============================
    with col1:
        scat = df.dropna(subset=["q_buy_exec", "yes_bid", "outcome"]).copy()
        liq_mask = (
            (scat["yes_spread"] <= max_spread)
            & (scat["yes_bid_size"] >= min_size)
            & (scat["yes_ask_size"] >= min_size)
        )
        scat["category"] = np.where(
            ~liq_mask, "ilíquido (gris)",
            np.where(scat["outcome"] >= 0.5,
                     "líquido — bin liquidó YES (rojo)",
                     "líquido — bin NO liquidó (verde)")
        )
        color_map = {
            "ilíquido (gris)": "#bcbcbc",
            "líquido — bin NO liquidó (verde)": "#138a08",
            "líquido — bin liquidó YES (rojo)": "#c0392b",
        }
        fig = px.scatter(
            scat.sort_values("category"),
            x="q_buy_exec", y="yes_bid", color="category",
            color_discrete_map=color_map, opacity=0.65,
            hover_data=["event_ticker", "subtitle"]
                if "subtitle" in scat.columns else ["event_ticker"],
            title="C) Edge: ¿Kalshi paga más de lo que cuesta cubrirse?",
        )
        # Zona verde "EDGE SELL YES" (yes_bid > q_buy_exec): triángulo
        # superior-izquierda por encima de y=x.
        fig.add_shape(
            type="path",
            path="M 0,0 L 0,0.65 L 0.65,0.65 Z",
            fillcolor="rgba(20, 180, 20, 0.10)",
            line=dict(width=0),
            layer="below",
        )
        # Diagonal y = x (la frontera donde no hay edge)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="black", dash="dash", width=1.5))
        fig.add_annotation(
            x=0.08, y=0.55, showarrow=False,
            text="<b>ZONA CON EDGE</b><br>(SELL YES rentable)",
            font=dict(size=11, color="#0a6608"),
            bgcolor="rgba(255,255,255,0.7)",
        )
        fig.add_annotation(
            x=0.50, y=0.18, showarrow=False,
            text="zona sin edge<br>(Kalshi caro)",
            font=dict(size=10, color="#888"),
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="black")),
                          selector=dict(name="ilíquido (gris)"))
        fig.update_traces(marker=dict(size=11, line=dict(width=0.6, color="black")),
                          selector=dict(name="líquido — bin NO liquidó (verde)"))
        fig.update_traces(marker=dict(size=11, line=dict(width=0.6, color="black")),
                          selector=dict(name="líquido — bin liquidó YES (rojo)"))
        fig.update_layout(height=430,
                          xaxis_range=[0, 0.65], yaxis_range=[0, 0.65],
                          xaxis_title="q_buy_exec  =  costo de replicar largo en Deribit",
                          yaxis_title="yes_bid Kalshi  =  precio al que vendes YES",
                          legend=dict(orientation="h", y=-0.18))
        _chart(fig)
        st.markdown(
            """
            **Qué ves:** cada punto es un bin × snapshot. Coordenadas:
            *eje X* = lo que cuesta cubrirte en Deribit; *eje Y* = lo que Kalshi
            te dejaría cobrar al vender YES.

            **Cómo interpretar:**
            - **Zona verde** (encima de y=x): Kalshi paga MÁS de lo que cuesta
              cubrirse → ARB SELL YES rentable.
            - **Zona blanca** (debajo de y=x): Kalshi paga MENOS → no hay edge.
            - **Puntos grises** = bins ilíquidos (no se pueden tradear de verdad).
            - **Puntos verdes grandes** = trades del slice donde el bin terminó
              **NO** (cobras tu yes_bid íntegro = ganas).
            - **Puntos rojos grandes** = bins del slice donde terminó **YES**
              (pagas $1, pierdes 1−yes_bid).

            **Diagnóstico:** queremos ver muchos puntos coloreados en la zona
            verde, mayoría verdes y pocos rojos.
            """
        )

    # ============================== D ==============================
    with col2:
        if pnl["n_trades"] > 0:
            t = pnl["trades"].copy()
            t["t_idx"] = np.arange(1, len(t) + 1)
            fig = px.line(t, x="t_idx", y="cum_pnl", markers=True,
                          hover_data=["event_ticker", "yes_bid",
                                      "q_buy_exec", "outcome"],
                          title=f"D) PnL realizado de la estrategia (th={threshold:.1%})")
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(height=430,
                              xaxis_title="# trade (orden cronológico)",
                              yaxis_title="PnL acumulado  [USD por contrato]")
            _chart(fig)
            st.markdown(
                f"""
                **Qué ves:** lo que habrías ganado/perdido si hubieras vendido
                YES en cada oportunidad del slice (con tus filtros actuales) en
                el orden en que aparecieron en el tiempo.

                **Cómo interpretar:**
                - Si la curva sube → la estrategia gana de media.
                - Si oscila plana o baja → no hay edge real, o se lo come la
                  varianza.
                - Saltos hacia abajo = trades donde el bin liquidó YES (perdiste).

                **Métricas actuales con tus filtros:**
                · n_trades = **{pnl['n_trades']}**
                · PnL = **${pnl['pnl_total']:+.2f}**
                · avg = **${pnl['avg_pnl']:+.4f}/trade**
                · winrate = **{pnl['winrate']:.1%}**
                · {pnl['n_events']} eventos distintos.

                **Caveats:** sin fees Kalshi (~$0.07/contrato), muestra pequeña.
                Necesitamos ~150+ trades antes de creérnoslo.
                """
            )
        else:
            st.info(
                "No hay trades con tus filtros actuales. Prueba a "
                "**relajar los sliders** del sidebar (subir `max_spread`, "
                "bajar `min_size` o `threshold`) y verás aparecer trades aquí."
            )


def render_backtest_interactive(df: pd.DataFrame, max_spread: float,
                                 min_size: float, threshold: float,
                                 fee_deribit: float, one_per_event: bool,
                                 event_pick: str = "first",
                                 model: str = "repl") -> None:
    st.subheader("Backtest interactivo")

    if df.empty:
        st.warning("No hay backtest CSV.")
        return

    liq = filter_liquid(df, max_spread, min_size)
    pnl = compute_pnl_rows(liq, threshold, fee_deribit=fee_deribit,
                           one_per_event=one_per_event,
                           event_pick=event_pick, model=model)

    # Comparativa lado a lado: best/first/random para que se vea el efecto
    if one_per_event:
        st.markdown("**Comparativa por modo de pick** (mismo threshold y filtros):")
        comp_rows = []
        for pick in ["best", "first", "random"]:
            p = compute_pnl_rows(liq, threshold, fee_deribit=fee_deribit,
                                 one_per_event=True, event_pick=pick, model=model)
            comp_rows.append({
                "pick": pick,
                "n_trades": p["n_trades"],
                "PnL_gross": p.get("pnl_gross_total", 0),
                "fees": p.get("fees_total", 0),
                "PnL_net": p["pnl_total"],
                "avg/trade": p.get("avg_pnl", 0) if p["n_trades"] else 0,
                "winrate": p.get("winrate", 0),
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, hide_index=True)
        st.caption("`best` es retrospectivo y optimista. `first` es timing realista. "
                   "Si la diferencia es grande, el edge depende de timing perfecto = no es real.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Filas tras filtro", f"{len(liq):,}")
    c2.metric("Trades", pnl["n_trades"],
              help="Max 1/evento" if one_per_event else "Todos")
    c3.metric("PnL bruto", f"${pnl.get('pnl_gross_total', 0):+.2f}")
    c4.metric("Fees totales", f"${pnl.get('fees_total', 0):.2f}")
    c5.metric("PnL NETO", f"${pnl['pnl_total']:+.2f}",
              delta=f"{pnl.get('avg_pnl', 0)*100:+.2f}c/trade"
                    if pnl["n_trades"] > 0 else None)
    c6.metric("Winrate", f"{pnl.get('winrate', 0):.1%}")

    if pnl["n_trades"] == 0:
        st.info("Sin trades. Ajusta filtros en la sidebar.")
        return

    st.markdown("**PnL acumulado a lo largo de los trades**")
    t = pnl["trades"].copy()
    t["t_idx"] = np.arange(1, len(t) + 1)
    fig = px.line(t, x="t_idx", y="cum_pnl", markers=True,
                  color="event_ticker",
                  hover_data=["snap_ts", "yes_bid", "q_buy_exec",
                              "outcome", "yes_spread", "yes_bid_size"])
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=420)
    _chart(fig)

    st.markdown("**Trades del slice (ordenados cronológicamente)**")
    show_cols = ["snap_ts", "event_ticker", "ticker", "subtitle",
                 "lower", "upper", "yes_bid", "yes_ask", "yes_spread",
                 "yes_bid_size", "q_buy_exec", "outcome", "pnl"]
    show_cols = [c for c in show_cols if c in t.columns]
    st.dataframe(t[show_cols], width="stretch", height=320)

    st.markdown("**PnL por evento**")
    by_event = (t.groupby("event_ticker")
                  .agg(n=("pnl", "size"),
                       pnl=("pnl", "sum"),
                       wins=("outcome", lambda x: (x < 0.5).sum()))
                  .reset_index().sort_values("pnl", ascending=False))
    by_event["winrate"] = by_event["wins"] / by_event["n"]
    fig2 = px.bar(by_event, x="event_ticker", y="pnl", color="pnl",
                  color_continuous_scale=["firebrick", "white", "forestgreen"],
                  hover_data=["n", "winrate"],
                  title="PnL por evento (verde = positivo)")
    fig2.update_layout(height=320)
    _chart(fig2)


def render_live(threshold: float, max_spread: float, min_size: float,
                fee_deribit: float = 0.015) -> None:
    st.subheader("Snapshot vivo  +  oportunidades del slice ahora")

    pipe = load_latest_snapshot_pipeline()
    if pipe is None:
        st.error("No hay snapshots en disco.")
        return
    if "error" in pipe:
        st.error(f"pipeline error: {pipe['error']}")
        return

    snap_ts = pipe["snap_ts"]
    age_s = (pd.Timestamp.utcnow() - snap_ts).total_seconds()
    color = "🟢" if age_s < 90 else ("🟡" if age_s < 300 else "🔴")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{color} Edad snapshot", fmt_age(age_s))
    c2.metric("Evento", pipe["event_ticker"])
    c3.metric("T_K (a settle)",
              fmt_age(pipe["T_K"] * 365 * 24 * 3600))
    c4.metric("Forward (PCP)", f"${pipe['F']:,.0f}")

    ann = pipe["annotated"].copy()

    # === Oportunidades AHORA ===
    st.markdown("### Oportunidades AHORA en el slice ultra-líquido")
    if "q_buy_exec" not in ann.columns:
        st.warning("snapshot pipeline no produjo q_buy_exec (¿pre-Fase 3.5?)")
    else:
        ann["yes_spread"] = ann["yes_ask"] - ann["yes_bid"]
        liq_mask = (
            (ann["yes_spread"] <= max_spread)
            & (ann["yes_bid_size"].fillna(0) >= min_size)
            & (ann["yes_ask_size"].fillna(0) >= min_size)
        )
        ann["edge_sell_yes_exec"] = ann["yes_bid"] - ann["q_buy_exec"]
        ann["fee_kalshi"] = ann["yes_bid"].apply(kalshi_fee)
        ann["fee_total"] = ann["fee_kalshi"] + fee_deribit
        ann["edge_net"] = ann["edge_sell_yes_exec"] - ann["fee_total"]
        opps = ann[liq_mask & (ann["edge_net"] > threshold)].copy()
        opps = opps.sort_values("edge_net", ascending=False)

        if opps.empty:
            st.info(f"No hay oportunidades NETAS de fees con threshold={threshold:.1%} "
                    f"en este snapshot. (El edge bruto puede existir pero el fee se lo come.)")
        else:
            st.success(f"{len(opps)} bin(s) con edge NETO SELL YES ejecutable.")
            cols = ["subtitle", "yes_bid", "yes_ask", "yes_spread",
                    "yes_bid_size", "q_buy_exec", "edge_sell_yes_exec",
                    "fee_total", "edge_net", "q_deribit"]
            cols = [c for c in cols if c in opps.columns]
            st.dataframe(opps[cols].head(20))

    # === Plot Q vs Kalshi por bin ===
    st.markdown("### Q Deribit vs Kalshi por bin (snapshot actual)")
    rel = ann[(ann["q_deribit"].fillna(0) > 0.005) | (ann["yes_mid"].fillna(0) > 0.02)].copy()
    rel["mid_strike"] = ((rel["lower"].fillna(rel["upper"]) + rel["upper"].fillna(rel["lower"])) / 2.0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Q Deribit", x=rel["mid_strike"],
                         y=rel["q_deribit"], marker_color="#d97706",
                         opacity=0.85, offset=-40, width=80))
    if "yes_mid" in rel.columns:
        fig.add_trace(go.Bar(name="Kalshi yes_mid", x=rel["mid_strike"],
                             y=rel["yes_mid"], marker_color="#2563eb",
                             opacity=0.6, offset=40, width=80))
    if "yes_bid" in rel.columns and "yes_ask" in rel.columns:
        fig.add_trace(go.Scatter(name="yes_bid", x=rel["mid_strike"], y=rel["yes_bid"],
                                 mode="markers", marker=dict(symbol="triangle-up", size=8, color="green")))
        fig.add_trace(go.Scatter(name="yes_ask", x=rel["mid_strike"], y=rel["yes_ask"],
                                 mode="markers", marker=dict(symbol="triangle-down", size=8, color="red")))
    fig.add_vline(x=pipe["F"], line_dash="dash", line_color="gray",
                  annotation_text=f"F={pipe['F']:,.0f}")
    fig.update_layout(height=430, barmode="overlay",
                      xaxis_title="Strike midpoint (USD)",
                      yaxis_title="Probabilidad")
    _chart(fig)

    # === SVI smile ===
    st.markdown("### Smile Deribit + fit SVI")
    fit_df = pipe["fit_df"]
    from src.model.svi import implied_vol_from_params
    k_grid = np.linspace(fit_df["k"].min() - 0.05, fit_df["k"].max() + 0.05, 300)
    iv_grid = implied_vol_from_params(k_grid, pipe["svi_params"], pipe["T_D"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fit_df["k"], y=fit_df["iv"], mode="markers",
                              marker=dict(size=8, color="#2563eb"),
                              name="market IV (OTM)"))
    fig2.add_trace(go.Scatter(x=k_grid, y=iv_grid, mode="lines",
                              line=dict(color="black", width=2), name="SVI fit"))
    fig2.update_layout(height=340, xaxis_title="log-moneyness k = ln(K/F)",
                       yaxis_title="implied vol")
    _chart(fig2)


def render_calibration(df: pd.DataFrame, diag: pd.DataFrame) -> None:
    st.subheader("Calibración y sanity")

    if df.empty:
        st.warning("No hay backtest CSV.")
        return

    # Reliability diagram (q_deribit vs realized) ----------------------
    st.markdown("### Reliability diagram — q_deribit vs realización")
    edges = np.linspace(0, 1, 11)
    df = df.copy()
    df["bucket"] = pd.cut(df["q_deribit"], bins=edges, include_lowest=True)
    rel = df.groupby("bucket", observed=True).agg(
        n=("outcome", "size"),
        mean_pred=("q_deribit", "mean"),
        mean_outcome=("outcome", "mean"),
    ).reset_index().dropna(subset=["mean_pred"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="black"),
                             name="calibración perfecta"))
    fig.add_trace(go.Scatter(
        x=rel["mean_pred"], y=rel["mean_outcome"],
        mode="markers+lines",
        marker=dict(size=rel["n"].clip(8, 30), color="#2563eb", opacity=0.85,
                    sizemode="area", sizeref=2.0 * rel["n"].max() / 900,
                    line=dict(width=1, color="black")),
        name="q_deribit",
        hovertemplate="bucket pred ≈ %{x:.3f}<br>realizado ≈ %{y:.3f}<br>n=%{text}",
        text=rel["n"],
    ))
    fig.update_layout(height=420, xaxis_title="probabilidad predicha",
                      yaxis_title="frecuencia realizada",
                      xaxis=dict(range=[-0.02, 1.02]),
                      yaxis=dict(range=[-0.02, 1.02]))
    _chart(fig)
    st.caption("Tamaño del punto = nº de filas en ese bucket. "
               "Cuanto más cerca de la diagonal, mejor calibrado el modelo.")

    # Brier por horizonte ---------------------------------------------
    st.markdown("### Brier score por horizonte T_K")
    YEAR_MIN = 365 * 24 * 60.0
    df["T_K_min"] = df["T_K"] * YEAR_MIN
    edges_min = [0, 5, 15, 30, 60, np.inf]
    df["horizon"] = pd.cut(df["T_K_min"], bins=edges_min, right=False)
    hor = df.groupby("horizon", observed=True)[["q_deribit", "yes_mid", "outcome"]].apply(
        lambda g: pd.Series({
            "n": len(g),
            "brier_q": ((g["q_deribit"].clip(1e-6, 1-1e-6) - g["outcome"]) ** 2).mean(),
            "brier_yesmid": ((g["yes_mid"].fillna(0).clip(1e-6, 1-1e-6) - g["outcome"]) ** 2).mean()
                if "yes_mid" in g.columns else np.nan,
            "p_yes": g["outcome"].mean(),
        })
    ).reset_index()
    hor["horizon_str"] = hor["horizon"].astype(str)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Brier q_deribit", x=hor["horizon_str"], y=hor["brier_q"],
                          marker_color="#1a8a1a"))
    fig2.add_trace(go.Bar(name="Brier yes_mid Kalshi", x=hor["horizon_str"], y=hor["brier_yesmid"],
                          marker_color="#2563eb"))
    fig2.update_layout(height=320, barmode="group",
                       xaxis_title="horizonte T_K (minutos)",
                       yaxis_title="Brier score (menor = mejor)")
    _chart(fig2)

    # Sanity sum_q ----------------------------------------------------
    if not diag.empty and "sum_q_deribit" in diag.columns:
        st.markdown("### Sanity: ∑Q por snapshot (debería ≈ 1)")
        fig3 = px.histogram(diag, x="sum_q_deribit", nbins=50,
                            range_x=[0.95, 1.05])
        fig3.add_vline(x=1.0, line_dash="dash", line_color="firebrick")
        fig3.update_layout(height=280)
        _chart(fig3)
        bad = diag[(diag["sum_q_deribit"] - 1).abs() > 0.01]
        if not bad.empty:
            st.warning(f"{len(bad)} snapshots con |∑Q − 1| > 0.01")
            st.dataframe(bad, width="stretch", height=200)
        else:
            st.success(f"Todos los {len(diag)} snapshots tienen ∑Q dentro de ±0.01.")


# --- main -------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Cross-Market Arb Analytics",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("Cross-Market Arbitrage  ·  Kalshi vs Deribit")

    # Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.header("Filtros de tradabilidad")
        threshold = st.slider("threshold edge (yes_bid − q_buy_exec)",
                              min_value=0.0, max_value=0.10,
                              value=0.01, step=0.005, format="%.3f")
        max_spread = st.slider("max_spread Kalshi (yes_ask − yes_bid)",
                               min_value=0.01, max_value=0.50,
                               value=0.03, step=0.005, format="%.3f")
        min_size = st.slider("min_size en bid y ask (contratos)",
                             min_value=0, max_value=500, value=50, step=10)

        st.divider()
        st.header("Fees (PnL neto)")
        st.caption("Kalshi: ceil(0.07·p·(1−p)) USD/contrato (oficial). "
                   "Deribit: aproximacion plana ajustable.")
        fee_deribit = st.slider("fee Deribit constante (USD/contrato)",
                                min_value=0.0, max_value=0.05,
                                value=0.015, step=0.005, format="%.3f")
        fee_multiplier = st.slider(
            "stress test: multiplicador de fees",
            min_value=1.0, max_value=4.0, value=1.0, step=0.5,
            help="Multiplica fees Kalshi+Deribit para ver robustez."
        )

        st.divider()
        st.header("Modelo de fair value")
        model_label = st.radio(
            "Q usada para edge",
            options=["repl", "exec"],
            format_func=lambda x: {
                "exec": "exec (SVI bid/ask teorica) — Sprint 1",
                "repl": "repl (replicacion REAL strikes Deribit) — Sprint 1.5",
            }[x],
            index=0,
            help="repl: solo bins con bracket Deribit cotizado, mas honesto. "
                 "exec: todos los bins, mas optimista.",
        )

        st.divider()
        st.header("De-correlacion")
        one_per_event = st.toggle(
            "Max 1 trade por evento", value=True,
            help="Snapshots consecutivos del mismo evento estan correlacionados.")
        event_pick = st.radio(
            "Como elegir el trade del evento",
            options=["first", "best", "random"],
            format_func=lambda x: {
                "first": "first opportunity — REALISTA (timing real)",
                "best": "best score — OPTIMISTA (cherry-pick retrospectivo)",
                "random": "random — intermedio",
            }[x],
            index=0,
            help="best: asume que sabes a priori cual snapshot dara mejor edge "
                 "(no es realista). first: vendes en cuanto aparece edge.",
        )

        st.divider()
        st.header("Fuente de datos")
        source_key = st.radio(
            "Backtest data source",
            options=["original", "event"],
            format_func=lambda x: _SOURCE_META[x]["label"],
            index=0,
        )
        meta = _SOURCE_META[source_key]
        csv_ready = csv_has_data(meta["csv"])
        if not csv_ready:
            st.warning(
                f"CSV vacío o no existe.  \nPuedes generarlo con:  \n`{meta['gen_cmd']}`"
            )
            if st.button("Generar backtest ahora", type="primary"):
                with st.spinner("Generando backtest... puede tardar un poco."):
                    result = run_backtest_from_app(meta["args"])
                if result.returncode == 0:
                    load_backtest.clear()
                    st.success("Backtest generado. Recargando dashboard...")
                    st.rerun()
                else:
                    st.error("El backtest falló.")
                    output = (result.stderr or result.stdout or "").strip()
                    if output:
                        st.code(output[-4000:])

        st.divider()
        if st.button("Recargar CSV (forzar)"):
            load_backtest.clear()
            load_latest_snapshot_pipeline.clear()
            st.rerun()
        st.caption("CSV se cachea por defecto. Pulsa para releer del disco.")

        st.divider()
        st.caption(
            "Backtest CSV: "
            + (str(meta["csv"]) if meta["csv"].exists() else "⚠ no existe")
        )
        st.caption(
            f"Snapshots: {len(list_snapshots(meta['snap_dir']))} en disco"
        )

    df, diag = load_backtest(str(meta["csv"]), str(meta["diag"]))
    fee_deribit_eff = fee_deribit * fee_multiplier

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Backtest interactivo",
                                       "Live snapshot", "Calibración"])
    with tab1:
        render_overview(df, max_spread, min_size, threshold,
                        fee_deribit_eff, one_per_event, fee_multiplier,
                        event_pick, model_label, snap_dir=meta["snap_dir"])
    with tab2:
        render_backtest_interactive(df, max_spread, min_size, threshold,
                                    fee_deribit_eff, one_per_event,
                                    event_pick, model_label)
    with tab3:
        render_live(threshold, max_spread, min_size, fee_deribit_eff)
    with tab4:
        render_calibration(df, diag)


if __name__ == "__main__":
    main()
