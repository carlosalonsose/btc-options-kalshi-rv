# Checkpoint del proyecto — explicado en plain language

> Este documento explica el proyecto en lenguaje sencillo, sin jerga, con
> ejemplos numéricos. Está pensado para releer cuando te pierdas.
> Para detalles técnicos: ver [README.md](README.md).

---

## 1. ¿Qué problema queremos resolver?

Hay dos sitios distintos donde se puede apostar al precio de Bitcoin:

**Kalshi**
- Vende "fichas" tipo: *"esta ficha paga $1 si BTC está entre $80,800 y
  $80,899.99 a las 14:00 UTC"*.
- Cada ficha cuesta entre $0 y $1 — el precio refleja la probabilidad
  estimada por el mercado.
- Para cada hora del día publican ~200 fichas cubriendo todos los rangos
  posibles de precio.

**Deribit**
- Vende **opciones clásicas** de BTC (calls y puts a distintos strikes y
  vencimientos).
- A partir de los precios de muchas opciones se puede *deducir* la
  distribución de probabilidad que el mercado le asigna al precio futuro
  de BTC.

**La idea:**
Si Deribit y Kalshi están de acuerdo en que la probabilidad de que BTC
acabe en `[80,800, 80,899.99]` es 25%, entonces la ficha Kalshi debería
costar $0.25. Si Kalshi la cotiza a $0.30 y Deribit dice 25%, alguien
está mal y hay 5 cents de margen — eso es **arbitraje**.

Nuestro proyecto detecta y (eventualmente) opera esas discrepancias.

---

## 2. ¿Qué guarda el snapshotter?

Un proceso `.py` graba a disco un snapshot con:

- **Deribit**: lista de las ~3000 opciones BTC vivas con sus bid/ask, IV, etc.
- **Kalshi**: lista de las ~300 fichas KXBTC abiertas con sus bid/ask y sizes.

Hay dos colectores: el de tiempo fijo (cada 60s) y el event-driven (sondea
más a menudo, solo escribe cuando cambia el quote state). El dataset canónico
del backtest usa el event-driven: ~8.9K snapshots sobre ~5 días. Cada snapshot
es ~190 KB comprimido (`*.json.gz`).

Esto es la **materia prima** del proyecto. Una vez en disco no necesitamos
más Internet para analizar el pasado.

---

## 3. ¿Qué hace el "pipeline" (`run_pipeline`)?

Es la función central. Tomas UN snapshot y te devuelve, para cada ficha
Kalshi del evento que toca, una **probabilidad teórica Q** según Deribit.

Pasos por dentro (lo que ocurre cuando le pasas un snapshot):

1. **Elegir un evento Kalshi.** Por defecto el más cercano en el tiempo
   (la próxima hora). Ej: `KXBTC-26MAY0918` que liquida a las 18:05 UTC
   del 9 de mayo.
2. **Calcular cuánto falta para que liquide.** Ej: `T_K = 35 minutos`.
3. **Elegir las opciones Deribit que vencen después de esa hora.** Ej:
   las que vencen mañana a las 08:00 UTC. `T_D = 14h`.
4. **Calcular el "forward" (precio futuro implícito).** Combinando calls
   y puts del mismo strike (put-call parity) sale F ≈ $80,815 cuando spot
   ≈ $80,808. Es lo que el mercado espera que valga BTC en T_D.
5. **Construir la "smile".** Para cada strike Deribit calculamos la
   *implied volatility* (cuán dispersa cree el mercado que va a ser la
   distribución). La smile es la curva de IVs en función del strike.
6. **Fittear una fórmula matemática (SVI)** a esa nube de puntos. SVI es
   simplemente una curva con 5 parámetros que captura bien la forma de la
   smile. Sirve para tener una IV continua en lugar de puntos sueltos.
7. **Calcular Q por ficha Kalshi.** Para cada rango `[L, U]` de la ficha,
   con la IV salida de la SVI, aplicas Black-Scholes (digital) para
   obtener `P(L ≤ S_T ≤ U)`. Esa es la `q_deribit`.
8. **Calcular "edges".** `edge_mid = yes_mid_kalshi - q_deribit`. Si > 0,
   Kalshi cotiza la ficha más cara que Deribit; si < 0, más barata.

Salida: un DataFrame con una fila por bin Kalshi, columnas con yes_bid,
yes_ask, q_deribit, etc.

---

## 4. ¿Qué añade Fase 3.5 al pipeline?

El problema de la Q teórica: usa el **precio medio** de cada opción Deribit.
Pero en la realidad, si quieres operar tienes que cruzar el bid-ask:

- Si COMPRAS una opción → pagas el ASK (caro).
- Si VENDES una opción → cobras el BID (más barato).

Fase 3.5 calcula **dos curvas más** además de la mid:

- `svi_bid`: usa los precios bid (peor caso si vendes en Deribit).
- `svi_ask`: usa los precios ask (peor caso si compras en Deribit).

Para cada bin Kalshi te da:
- `q_buy_exec`  = lo que **pagas** si quieres comprar la replicación en Deribit.
- `q_sell_exec` = lo que **cobras** si vendes la replicación.
- `q_deribit`   = la mid teórica (referencia).

Siempre `q_sell_exec ≤ q_deribit ≤ q_buy_exec`.

Edges ejecutables (los que cuentan):
- `edge_buy_yes_exec  = q_sell_exec − yes_ask_kalshi` >0 ⇒ rentable comprar YES.
- `edge_sell_yes_exec = yes_bid_kalshi − q_buy_exec` >0 ⇒ rentable vender YES.

---

## 5. ¿Qué hace exactamente el backtest? (tu pregunta concreta)

El pipeline anterior te dice "cuánto debería costar la ficha hoy". El
backtest responde a una pregunta distinta:

> *Si hubiera operado siguiendo este modelo en el pasado, ¿cuánto habría
> ganado o perdido?*

Pasos del backtest (`src/runner/backtest.py`):

### 5.1. Cargar el outcome real

Primero, una sola vez al arrancar:
1. Llama al endpoint `/markets?status=settled&series_ticker=KXBTC` de Kalshi.
2. Recoge **todos los mercados que ya liquidaron** en la ventana del
   backtest. Cada uno trae `result = "yes"` o `"no"` (si BTC acabó en
   ese rango o no).
3. Cachea esto en disco (`data/kalshi_settled.json.gz`) para no volver a
   pedirlo.

Para nuestra ventana actual: ~5,126 mercados settled.

### 5.2. Iterar sobre los snapshots

La corrida canónica usa `--stride 1` sobre el dataset event-driven completo
(~8.9K snapshots, ~9 min). Para iterar rápido se puede subir el stride.

Para cada snapshot:
1. **Corre el pipeline completo** sobre ese snapshot. Le da una `q_deribit`,
   `q_buy_exec`, `q_sell_exec` por bin del evento más cercano en ese instante.
2. **Para cada bin**, busca su ticker en la cache de settled markets.
3. Si el ticker existe en el cache (= ese bin ya liquidó):
   - Obtiene el outcome real (`yes`=1, `no`=0).
   - Guarda una fila con: snapshot timestamp, ticker, lower, upper,
     yes_bid, yes_ask, yes_spread, q_deribit, q_buy_exec, q_sell_exec,
     outcome (0/1), T_K, etc.

Resultado: un CSV grande con **1,379,512 filas** (71 eventos liquidados,
7,775 snapshots con outcome). Cada fila = una predicción evaluable contra su
outcome real.

### 5.3. Calcular métricas agregadas

Sobre ese CSV:

**A. Brier score** (mide calidad probabilística):
```
brier = mean((q_predicha - outcome_real)²)
```
Más bajo = mejor.
- q_deribit: 0.004812
- yes_mid Kalshi: 0.006023
→ El modelo q es ~20% mejor que los mids de Kalshi en Brier.

**B. Log loss**: similar a Brier pero penaliza más las "convicciones equivocadas".
- q_deribit: 0.018406
- yes_mid Kalshi: 0.028052
→ ~34% mejor.

**Matiz importante:** la tasa base de YES es 0.56%. ~98% de las observaciones
caen en el bucket 0-10%, así que ambos modelos puntúan bien simplemente
prediciendo ≈0. La mejora es real pero su magnitud está inflada por la tasa
base, no por habilidad en la región de probabilidad alta.

**C. Reliability diagram (calibración)**: agrupa las predicciones en 10
buckets (0-10%, ..., 90-100%) y mira si "cuando el modelo dijo 30%, de verdad
liquidó YES en ~30% de los casos". q_deribit está bien calibrado hasta ~0.5;
por encima la muestra es escasa (ruido, no señal).

**D. Reliability por horizonte T_K**: ¿predice igual de bien a 5 min que a
60 min? Brier sube de 0.0034 (0-5 min) a 0.0071 (>60 min): se degrada al
alargar el horizonte. VLT scaling aguanta mejor cerca del settle.

### 5.4. Calcular PnL ejecutable (la parte clave)

La pregunta: *"si hubiera tradeado siguiendo el modelo, cuánto habría ganado?"*

Estrategia simulada:
- Para cada fila del CSV con `yes_bid_kalshi - q_buy_exec > threshold`:
  ```
  Acción: VENDER YES Kalshi a yes_bid + comprar replicación Deribit
  PnL    = yes_bid - outcome_real
  ```
  Es decir, cobras `yes_bid` por adelantado, y al settle pagas `1` si el bin
  liquidó YES (perdiste) o `0` si liquidó NO (ganaste todo el bid).

Por defecto suma todos los trades sin filtros. Y también con un *grid de
liquidez*: filtra solo a bins con spread bajo y profundidad alta antes de
contar trades.

Lo que vimos (corrida canónica, replicación discreta, **fees descontados**,
1 trade por evento):
- PnL **negativo a todos los thresholds** salvo +$0.06 marginal al 5%
  (24 buy / 63 sell). Solo el 14% de las observaciones tiene un bracket
  Deribit ejecutable.
- Slices ultra-líquidos (spread ≤ 1-2c AND size ≥ 50) dan PnL pequeño
  positivo (+$2.3 a +$2.6 en ~40-60 trades), pero son sub-muestras
  pequeñas de 1.38M filas: "no descartado todavía", no edge confirmado.

### 5.5. Lo que el backtest NO hace todavía

- **Fees ya descontados** (Kalshi `ceil(0.07·p·(1−p))` + Deribit $0.015).
- **No simula slippage** al ejecutar volúmenes grandes en Deribit.
- **No modela "fill probability"** si la estrategia es postear limit orders
  en Kalshi en lugar de cruzar el spread.
- **No descuenta el coste del horizon mismatch** (Kalshi liquida en T_K,
  Deribit en T_D > T_K — la replicación no es perfecta).

---

## 6. Resumen visual del pipeline completo

```
SNAPSHOTTER (event-driven)
      │
      ▼
data/event_snapshots/2026-05-12/HHMMSS_micros.json.gz
      │
      ▼
  ┌──────────────── BACKTEST ────────────────┐
  │                                          │
  │   1. Para cada snapshot en disco:        │
  │      run_pipeline()                      │
  │        ↓                                 │
  │      bins con q_deribit, q_buy_exec,     │
  │      q_sell_exec, edge_*                 │
  │                                          │
  │   2. Cargar outcome real de Kalshi       │
  │      (cache settled markets)             │
  │                                          │
  │   3. Joinear snapshot × outcome          │
  │      → 1,379,512 filas                   │
  │                                          │
  │   4. Métricas: Brier, log loss,          │
  │      calibración, PnL ejecutable         │
  └───────────────────┬──────────────────────┘
                      │
                      ▼
              data/reports/backtest_canonical.csv
                      │
                      ▼
        STREAMLIT APP (analytics interactivo)
```

---

## 7. Qué hemos descubierto (resultados)

### El bueno
- **El modelo q puntúa mejor que los mids de Kalshi**: Brier ~20% mejor,
  log loss ~34% mejor (matiz: amplificado por la tasa base 0.56%).
- **Calibrado** hasta el bucket ~0.5; por encima, muestra escasa.

### El malo
- **Los spreads Kalshi son enormes**. El edge teórico se evapora al cruzar.
- **Solo 14% de las observaciones** tiene un bracket Deribit ejecutable.
- **PnL neto negativo** a todos los thresholds salvo +$0.06 marginal al 5%.

### Lo que sigue siendo prometedor (con cautela)
- Slices ultra-líquidos (spread ≤ 1-2c, size ≥ 50) dan PnL pequeño
  positivo (+$2.3 a +$2.6 en ~40-60 trades). Son sub-muestras pequeñas
  de 1.38M filas: no es edge confirmado, es "no descartado".

---

## 8. Estado actual

| Componente | Estado | Comentario |
|---|---|---|
| Snapshotter (60s + event-driven) | ✅ | dataset canónico = event-driven |
| Pipeline run_pipeline | ✅ | mid + bid/ask SVI fits |
| Backtest offline | ✅ | fees + grid de liquidez + repl. discreta |
| Visualización findings | ✅ | findings.png estático |
| Dashboard Tkinter | ✅ | snapshot actual (vigilancia) |
| Streamlit analytics | ✅ | histórico interactivo |
| Subset reproducible | ✅ | data/sample_snapshots/ committeado |
| Paper trading | ⏳ | solo si el edge persiste con más datos |

---

## 9. Cómo reproducir

```bash
# Reproducir un backtest sobre el subset committeado (sin datos externos):
bash backtest.sh --snapshots data/sample_snapshots --stride 1 \
                 --out data/reports/backtest_sample.csv

# Recolectar datos propios y correr el backtest completo:
bash event_start.sh                       # colector event-driven
bash backtest.sh --snapshots data/event_snapshots --stride 1

# Analytics interactivo:
bash analytics.sh
```

---

## 10. Glossary mínimo (las 6 palabras clave)

- **Bin / ficha Kalshi**: contrato binario que paga $1 si BTC está en un
  rango concreto al final de la hora.
- **Q (probabilidad neutral al riesgo)**: probabilidad implícita en los
  precios de Deribit de que BTC caiga en un rango.
- **Smile**: cómo varía la "implied volatility" con el strike de las
  opciones — captura cuán dispersa cree el mercado que será la distribución.
- **SVI**: una fórmula con 5 parámetros que ajusta bien la smile. Sirve
  para tener una IV continua.
- **VLT scaling**: asunción de que la varianza crece linealmente con el
  tiempo. Permite reusar la IV de T_D para calcular probabilidad a T_K < T_D.
- **Replicación ejecutable**: precio del bin construido cruzando los
  bid-ask reales de las opciones Deribit (no los mids). Es el coste real
  de cubrirse.
