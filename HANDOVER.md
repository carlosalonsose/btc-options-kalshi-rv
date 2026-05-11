# Cross-Market Arbitrage BTC: Kalshi vs Deribit — Handover

> Documento self-contained para continuar el proyecto en un chat nuevo.
> Pegar entero como primer mensaje, o pedir al modelo `cat HANDOVER.md`.

## 1. Sobre el usuario
- Project root: `/Users/carlosalonso/Cross_market_arbitrage`.
- Idioma de trabajo: español.
- No es CS student; conoce finanzas cuantitativas a nivel concepto, no implementación.
- Prefiere orientación clara y pasos concretos, no menús de opciones largos.
- Se pierde con jerga técnica sin contexto. Usar plain language en decisiones.
- Mac con Terminal app y zsh. Tuvo problemas históricos con copy-paste de comandos
  con comillas dobles (smart-quotes); ya hay scripts `.sh` para evitarlo.
- Destino del proyecto: pasar de notebook a un servicio `.py` que opere automáticamente.

## 2. Objetivo
Detectar y eventualmente operar discrepancias de precio entre:
- **Kalshi** — contratos binarios de rango sobre BTC (`KXBTC`) que pagan 1 si BTC liquida
  en `[L,U]`, 0 si no.
- **Deribit** — opciones vanilla BTC, de las que se extrae una distribución
  neutral al riesgo de BTC y se calcula `P(L ≤ S_T ≤ U)`.

Comparar precio Kalshi vs probabilidad implícita Deribit. Si difieren más que
costes/fricciones reales, hay señal explotable.

## 3. Hallazgos estructurales (lectura obligatoria — no re-debatir)
El framing original "extraer Q a expiry T desde Deribit y comparar con Kalshi al
mismo T" **no se realiza** con los mercados que existen:

- **Kalshi `KXBTC` solo tiene mercados horarios intradía**. Cada hora del día listan
  ~50–200 bins que cubren el strip de precios BTC. Settlement: media de
  60 segundos del **CF Benchmarks BRTI** antes de la hora.
- **Deribit solo vence a las 08:00 UTC** (daily / weekly / monthly). Settlement:
  media de 30 min del **Deribit BTC Index** antes de las 08:00 UTC.
- **Mismatch de índice**: BRTI ≠ Deribit Index (diferencia de pocos bps en horizontes
  cortos, calibrable).

**Decisión tomada:** opción "horizonte intradía" (la única que permite live trader).
Método requerido:
- SVI fiteado a la smile Deribit más corta disponible (T_D > T_K, típicamente
  next-day 08:00 UTC).
- **VLT scaling** (Variance Linear in Time): `σ_imp(K)` independiente de T.
  Bajo VLT, la `σ` extraída de Deribit T_D se reusa para calcular probabilidades a T_K.
- Q-bin via BS digital con esa `σ` y `T_K`.
- Convolución de basis BRTI–Deribit (deshabilitada por defecto, parámetro listo en
  `BasisModel`).

Otras opciones contempladas y descartadas para "live trader":
- `KXBTCY` anual: una sola observación, no es proyecto continuo.
- One-touch barrier `KXBTCMAXMON`/`MINMON`: matemáticamente otro proyecto
  (replicación con barrier statics, no verticales).

## 4. Estado del proyecto

| Fase | Estado | Descripción |
|---|---|---|
| 0 | ✅ | Scope decidido: horizonte horario |
| 1 | ✅ | Snapshotter `.py` grabando Deribit + Kalshi cada 60s |
| 2 | ✅ | Módulos puros (load, BS, SVI, intraday_q, edge) + `analyze.py` end-to-end |
| 2.5 | ✅ | Dashboard Tkinter sobre el pipeline |
| 3 | ✅ | Backtest offline: q_deribit calibrado, edge ejecutable < 0 (ver §13) |
| 3.5 | ⏳ | Bootstrap CI del SVI + replicación-precio ejecutable (load-bearing) |
| 4 | ⏳ | Live service: WebSocket + paper trading + risk + kill switches |
| 5 | ⏳ | Hardening producción: reconnects, idempotencia, alertas |

**Estado operacional:**
- Snapshotter persistente lanzado por el usuario via `bash start.sh`.
- 575+ snapshots ya en disco al cierre de Fase 2.5.

## 5. Estructura del repo

```
/Users/carlosalonso/Cross_market_arbitrage/
├── .venv/                         python 3.9 + numpy/scipy/pandas/matplotlib/requests
├── requirements.txt
├── start.sh                       lanza snapshotter persistente con nohup
├── stop.sh                        mata snapshotter
├── dash.sh                        lanza dashboard Tkinter
├── HANDOVER.md                    este documento
├── src/
│   ├── snapshotter.py             Fase 1: poll Deribit + Kalshi → data/snapshots/
│   ├── io/load.py                 carga snapshots a DataFrames (sin red)
│   ├── model/
│   │   ├── black_scholes.py       bs_call, bs_put, bs_digital_call, implied_vol
│   │   ├── svi.py                 SVIParams, fit_svi, total_variance(_and_derivs),
│   │   │                          forward_from_pcp, implied_vol_from_params
│   │   └── intraday_q.py          range_prob_intraday, range_prob_open_ended,
│   │                              BasisModel
│   ├── signal/edge.py             annotate_edges, basket_summary
│   ├── runner/
│   │   ├── analyze.py             run_pipeline() puro + run() CLI
│   │   └── backtest.py            Fase 3: itera snapshots, joinea con outcome
│   │                              real Kalshi settled → Brier, log loss,
│   │                              calibración, sweep PnL ejecutable
│   └── ui/dashboard.py            Tkinter dashboard sobre run_pipeline()
├── data/
│   ├── snapshots/                 YYYY-MM-DD/HHMMSS.json.gz (~190 KB)
│   │   └── snapshotter.log
│   ├── kalshi_settled.json.gz     cache settled markets para backtest
│   └── reports/                   plots opcionales + backtest CSV/log
└── stubs antiguos: derbit_client.py, kalshi_client.py, market_normalizer.py,
                    probability_engine.py, signal_engine.py, surface_engine.py
                    (placeholders del usuario; mapean a la arquitectura actual,
                    no se usan)
```

## 6. Schema de datos

**Snapshot** (`data/snapshots/.../*.json.gz`):
```json
{
  "snapshot_ts_utc": "2026-05-09T17:30:04Z",
  "schema_version": 1,
  "deribit": {
    "instruments_usdc": [...], "book_summary_usdc": [...],
    "instruments_btc":  [...], "book_summary_btc":  [...],
    "index_btc_usdc":   {...}, "index_btc_usd":     {...}
  },
  "kalshi": {
    "markets_open": [...],
    "events_open":  [...]
  }
}
```

**Schema Kalshi market actual** (lo que hay realmente en `markets_open[i]`):
- Precios con sufijo `_dollars` (ya en [0,1], NO dividir por 100):
  `yes_bid_dollars`, `yes_ask_dollars`, `no_bid_dollars`, `no_ask_dollars`,
  `last_price_dollars`.
- Sizes/volumen con sufijo `_fp`: `yes_bid_size_fp`, `yes_ask_size_fp`,
  `volume_fp`, `volume_24h_fp`, `open_interest_fp`.
- Rangos: `floor_strike` y `cap_strike` (uno puede ser null en bins-tail abiertos).
- `expected_expiration_time` (ISO 8601 UTC) para settlement.
- `subtitle`: human-readable bin (ej `"$80,300 to 80,399.99"`).
- `rules_primary`: regla de settlement detallada (mencionar si BRTI vs otro índice).

`io/load.py` ya hace el rename interno a nombres limpios: `yes_bid`, `yes_ask`,
`yes_mid`, `yes_spread`, `lower`, `upper`, etc.

## 7. Metodología matemática implementada

**Pipeline `run_pipeline(snap, event_ticker)`:**
1. Extraer eventos abiertos → elegir uno (default: settlement más cercano al snapshot).
2. Determinar `T_K = expected_expiration_time - snap_ts` en años.
3. Elegir Deribit expiry `T_D` posterior al settle Kalshi (la más cercana posterior).
4. Forward via put-call parity: `F = K + e^(rT)·(C - P)` mediano sobre pares ATM.
5. Construir smile OTM-only: puts para K<F, calls para K>=F → IV via `brentq`.
6. Fitear **SVI raw**: `w(k) = a + b·(ρ·(k-m) + √((k-m)² + σ²))` con
   `least_squares` ponderado por `1/spread`.
7. Para cada bin Kalshi `[L, U]` (manejando tails abiertos):
   - Bajo VLT: `σ_imp(K) = √(w(ln(K/F))/T_D)` evaluado en K.
   - `P^Q(S_T_K > K) = N(d2)` con esa σ y T_K.
   - `Q(L ≤ S_T_K ≤ U) = P(>L) - P(>U)`.
8. Edge:
   - `edge_mid       = yes_mid - q_deribit`
   - `edge_buy_yes   = q_deribit - yes_ask`  (>0 ⇒ comprar YES Kalshi)
   - `edge_sell_yes  = yes_bid  - q_deribit` (>0 ⇒ vender YES Kalshi)

**Validación cuantitativa:** `sum(q_deribit)` sobre el strip ≈ 0.9999 (Q normalizada).

**Derivadas analíticas SVI** ya implementadas en
`total_variance_and_derivs(k, p) → (w, w', w'')`. **No usadas todavía** por el
camino actual (que va punto a punto via BS digital). Disponibles para Fase 3.5
si pasamos a densidad RND analítica via Gatheral.

## 8. Decisiones técnicas tomadas (no re-debatir)

- **JSON.gz** para snapshots, no parquet → lossless, schema-flexible, sin pyarrow.
- **VLT scaling** como asunción de term structure → simple, defendible para horizontes
  cortos. Mejora futura: calibrar nivel σ con realized vol últimos 60 min y
  mantener la *forma* SVI para skew.
- **Smile OTM-only** (puts K<F + calls K>=F) → estándar, evita problemas numéricos
  con opciones ITM cerca de paridad.
- **SVI raw 5-param** con bounds:
  `[a∈[-1,1], b∈[1e-6,5], ρ∈[-0.999,0.999], m∈[-2,2], σ∈[1e-4,5]]`.
- **`BasisModel`** (BRTI − Deribit Index) definido como clase pero default
  `mu=0, sigma=0` → identidad. Calibrar con datos históricos en Fase 3.
- **Tasa r=0** asumida en BS y PCP (apropiado para horizontes cripto cortos).
- **Refactor**: `analyze.py` expone `run_pipeline(snap, event_ticker) -> dict`
  como función pura; `run()` es solo el CLI printer encima. Reusable desde la UI.

## 9. Cómo operar

```bash
# Todo desde /Users/carlosalonso/Cross_market_arbitrage

# === Snapshotter (proceso de larga duración) ===
bash start.sh           # arranca persistente (nohup), kill previo si existe
bash stop.sh            # mata
tail data/snapshots/snapshotter.log

# === Pipeline analítico (CLI) ===
.venv/bin/python -m src.runner.analyze
.venv/bin/python -m src.runner.analyze --plot
.venv/bin/python -m src.runner.analyze --event KXBTC-26MAY0917
.venv/bin/python -m src.runner.analyze --snapshot data/snapshots/2026-05-09/170000.json.gz

# === Dashboard Tkinter ===
bash dash.sh
# Auto-refresh cada 30s. Dropdown para cambiar evento. Plot smile + bars + tabla.

# === Backtest offline (Fase 3) ===
.venv/bin/python -m src.runner.backtest --stride 5     # 1 cada 5 snapshots
.venv/bin/python -m src.runner.backtest --refresh      # re-fetch settled
.venv/bin/python -m src.runner.backtest --limit 5      # smoke test
# Outputs:
#   data/reports/backtest.csv         filas (snap_ts, ticker, q, outcome, ...)
#   data/reports/backtest_diag.csv    diagnostico por snapshot
#   data/reports/backtest_report.txt  metricas formateadas (si pipeas con tee)
```

## 10. Issues conocidos / mejoras planificadas

**Fase 3 (backtest offline) — ✅ implementada en `src/runner/backtest.py`.**
Resultados resumidos en §13. Si quieres re-ejecutar con más datos o stride menor:
`python -m src.runner.backtest --stride 5`. La cache settled vive en
`data/kalshi_settled.json.gz`; usar `--refresh` para invalidar.

**Fase 3.5 (mejoras al modelo, antes de live):**
- **Bootstrap CI del SVI**: resamplear IVs por su bid-ask noise → IC del Q por bin
  → filtrar señales sub-CI. Crítico antes de operar de verdad.
- **Calibrar `BasisModel`** desde histórico BRTI vs Deribit Index. Necesita feed
  BRTI (CF Benchmarks o proxy con Coinbase BTC-USD). Snapshotter ya guarda
  `index_btc_usdc` y `index_btc_usd` por si sirve.
- **Replicación-precio ejecutable**: precio del bin via vertical spreads Deribit
  (asks para comprar la réplica, bids para venderla). Sustituye "Q vs P" por
  "P_repl_executable vs P_kalshi_executable", que es lo que de verdad se opera.
- **Densidad RND analítica** desde SVI usando `total_variance_and_derivs`
  (Gatheral closed form), no via BS digital strike-by-strike. Más rápido y
  más estable.
- **Durrleman / butterfly arb check**: verificar `g(k) ≥ 0` en el SVI fit.
  Crítico con DTE corto.
- **Gate de tradabilidad**: solo flagear bins con `edge > k · CI_width` y
  liquidez mínima en ambas patas.

**Fase 4 (live):**
- WebSocket Deribit (`public/subscribe` a `book.<instrument>` y similares) en vez
  de REST polling.
- Kalshi WebSocket para bid/ask updates en tiempo real.
- OMS con paper trading primero. Kalshi exchange API requiere auth.
- Kill switches (max drawdown, model staleness, market halts).
- Risk limits (max position per bin, max gross, max basis exposure).

## 11. Estilo de respuesta para el nuevo Claude

- Conciso, directo, en español.
- Si hay decisiones de scope grandes: **una recomendación clara**, no un menú.
- Si el usuario dice "estoy perdido" → reorientar con plain language, sin jerga.
- Si el usuario dice "ejecuta" → ejecutar; no proponer más opciones.
- Cuando ejecutes shell con copy-paste para el usuario: evitar comillas dobles
  complejas (causaron problemas de smart-quotes en su Mac);
  preferir scripts `.sh` que él invoca con `bash script.sh`.
- Es OK que respuestas técnicas sean detalladas si el usuario pide detalles,
  pero el comportamiento por defecto es brevedad orientativa.

## 12. Diagnóstico cuantitativo del último run válido (referencia)

Snapshot `2026-05-09 17:30 UTC`. Evento elegido: `KXBTC-26MAY0914`
(settle 18:05 UTC, T_K = 0.58h ≈ 35 min). Deribit smile: `2026-05-10 08:00 UTC`,
T_D = 14.5h. Spot $80,808, forward (PCP) $80,815.

SVI fit: 11 puntos OTM, success=True, cost ≈ 7e-13.

Distribución: masa concentrada en bins 80,700–80,899
(`q ≈ 0.55` agregada en esos dos bins). Kalshi `yes_mid` esos bins ≈ 0.485.
`sum(q_deribit) = 0.9999`.

Todos los `edge_buy_yes` y `edge_sell_yes` de bins relevantes son **negativos**:
el spread bid/ask de Kalshi se come la señal de mids. Confirma que la Fase 3.5
"replicación-precio ejecutable" es load-bearing antes de operar — los mids no
son tradables, hay que comparar contra ejecutables reales en ambos lados.

## 13. Resultados Fase 3 (backtest offline) — 2026-05-09

Datos: 575 snapshots cubriendo 2026-05-08 16:28 → 2026-05-09 17:30 UTC.
Backtest evaluado a `--stride 5` (≈1 snapshot cada 5 min) → 117 snapshots,
21 eventos horarios liquidados, **17 984 filas (bin × snapshot) con outcome real**.

**Métricas agregadas:**

| Métrica | q_deribit | yes_mid Kalshi |
|---|---|---|
| Brier score | **0.00417** | 0.00771 |
| Log loss | **0.01418** | 0.03505 |

q_deribit predice ~46% mejor que el mid de Kalshi en Brier, y >2× mejor en log
loss. `sum(q_deribit)` ∈ [0.9999, 1.0000] siempre — Q normalizada perfecta.

**Calibración q_deribit por bucket:**

| bucket | n | mean_pred | mean_outcome | bias |
|---|---|---|---|---|
| (0.0, 0.1] | 17 597 | 0.00092 | 0.00085 | -0.00007 |
| (0.1, 0.2] |   216  | 0.149   | 0.153   | +0.004 |
| (0.2, 0.3] |   118  | 0.246   | 0.263   | +0.016 |
| (0.3, 0.4] |    31  | 0.343   | 0.323   | -0.021 |
| (0.4, 0.5]+|    23  | (escasos) | (ruido) | — |

Excelente calibración hasta 0.4. Por encima de 0.4 hay solo 23 filas (porque
en cada evento solo 1-2 bins liquidan YES); buckets >0.4 son ruido muestral,
no señal.

**Calibración yes_mid Kalshi (peor):**

| bucket | n | mean_pred | mean_outcome | bias |
|---|---|---|---|---|
| (0.2, 0.3] | 462 | 0.250 | 0.043 | **-0.207** |
| (0.4, 0.5] | 242 | 0.450 | 0.062 | **-0.388** |

Los mids Kalshi sobreestiman sistemáticamente la prob real porque el spread es
ancho (bid bajo, ask alto, mid arriba de la realización). Esto es una propiedad
de la microestructura, no un fallo del exchange — los mids no son ejecutables.

**Reliability por horizonte T_K:** Brier q_deribit es estable ~0.0039–0.0044 desde
5 min hasta >60 min antes del settle. No se deteriora apreciablemente al alargar
T_K → SVI + VLT scaling aguantan en horizontes intradía cortos.

**Edge ejecutable (sweep de thresholds, 1 contrato/fila, sin fees):**

| threshold | n_buy | n_sell | buy_pnl | sell_pnl | total |
|---|---|---|---|---|---|
| 0.5%  | 148 | 166 | -6.58 | -10.40 | **-16.98** |
| 1.0%  | 129 | 147 | -6.51 |  -9.25 | **-15.76** |
| 2.0%  | 103 | 130 | -5.57 | -10.82 | **-16.39** |
| 5.0%  |  73 |  94 | -4.78 |  -9.34 | **-14.12** |
| 10.0% |  35 |  66 | -1.83 |  -7.69 |  **-9.52** |

**A todos los thresholds, la estrategia "comprar YES si q > yes_ask + th" /
"vender YES si yes_bid > q + th" pierde dinero.** Confirmado: el spread Kalshi
se come la señal, exactamente como predijo el HANDOVER. El modelo es bueno;
el ejecutable contra mids/quotes no.

**Implicación operacional:** Fase 3.5 "replicación-precio ejecutable" no es
opcional — es **prerequisito** para Fase 4. Concretamente:
- Comparar `P_repl_executable_Deribit` (precio del bin via vertical de calls)
  contra `yes_ask_Kalshi` y `yes_bid_Kalshi`. El edge real solo aparece cuando
  el spread Kalshi se desvía de un precio ejecutable cotizable, no del Q teórico.
- Filtrar señales con bootstrap CI del SVI (vendrá cargado de IV bid/ask noise).
- Solo flagear bins donde `edge > k · CI_width` Y haya size mínima en ambas patas.

**Caveats del backtest:**
- 21 eventos solo, en ventana corta (~25 h). Resultados son indicativos, no
  estadísticamente sólidos. Re-correr con más datos al acumular.
- Filas correlacionadas dentro de un mismo evento (4-12 snapshots por evento,
  un mismo bin → 4-12 filas con outcome idéntico). Brier y PnL están inflados
  en n efectivo, no en bias.
- PnL ejecutable ignora fees Kalshi (~$0.07/contrato típico) y el coste de
  ejecutar el lado Deribit (cruzar bid-ask de la opción).
