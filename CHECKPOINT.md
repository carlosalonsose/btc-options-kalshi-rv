# Checkpoint del proyecto — explicado en plain language

> Este documento explica el proyecto en lenguaje sencillo, sin jerga, con
> ejemplos numéricos. Está pensado para releer cuando te pierdas.
> Para detalles técnicos: ver [HANDOVER.md](HANDOVER.md).

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

Cada **60 segundos** un proceso `.py` graba a disco un snapshot con:

- **Deribit**: lista de las ~3000 opciones BTC vivas con sus bid/ask, IV, etc.
- **Kalshi**: lista de las ~300 fichas KXBTC abiertas con sus bid/ask y sizes.

Cada snapshot es ~190 KB comprimido (`*.json.gz`). En 24h se acumulan
~1,500 archivos. Llevamos ~25h grabando = ~575 archivos.

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

Por defecto con `--stride 5` evalúa 1 snapshot cada 5 minutos. Para nuestros
575 snapshots → 117 evaluaciones (~13 segundos en total).

Para cada snapshot:
1. **Corre el pipeline completo** sobre ese snapshot. Le da una `q_deribit`,
   `q_buy_exec`, `q_sell_exec` por bin del evento más cercano en ese instante.
2. **Para cada bin**, busca su ticker en la cache de settled markets.
3. Si el ticker existe en el cache (= ese bin ya liquidó):
   - Obtiene el outcome real (`yes`=1, `no`=0).
   - Guarda una fila con: snapshot timestamp, ticker, lower, upper,
     yes_bid, yes_ask, yes_spread, q_deribit, q_buy_exec, q_sell_exec,
     outcome (0/1), T_K, etc.

Resultado: un CSV grande (`data/reports/backtest.csv`) con **17,984 filas**.
Cada fila = una predicción que vamos a poder evaluar contra su outcome real.

### 5.3. Calcular métricas agregadas

Sobre ese CSV:

**A. Brier score** (mide calidad probabilística):
```
brier = mean((q_predicha - outcome_real)²)
```
Más bajo = mejor.
- q_deribit: 0.00417
- yes_mid Kalshi: 0.00771
→ El modelo q es ~46% mejor que los mids de Kalshi.

**B. Log loss**: similar a Brier pero penaliza más las "convicciones equivocadas".
- q_deribit: 0.0142
- yes_mid Kalshi: 0.0351

**C. Reliability diagram (calibración)**: agrupa las predicciones en 10
buckets (0-10%, 10-20%, ..., 90-100%) y mira si "cuando el modelo dijo 30%,
de verdad liquidó YES en ~30% de los casos". Resultado: q_deribit está
**casi perfectamente calibrado** hasta el bucket 0.4.

**D. Reliability por horizonte T_K**: ¿el modelo predice mejor cuando faltan
5 min al settle vs 60 min? Resultado: estable, prácticamente igual de bueno
en todos los horizontes (0.0039 → 0.0044).

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

Lo que vimos:
- **Sin filtrar:** PnL siempre **negativo** a todos los thresholds.
- **Con filtro slice ultra-líquido** (spread ≤ 3c AND bid_size ≥ 50):
  PnL **positivo** (~+$1.67 acumulado en 18 trades).

### 5.5. Lo que el backtest NO hace todavía

- **No descuenta fees** Kalshi (~$0.02/contrato) ni Deribit (~$0.01/contrato).
  El edge bruto $0.09/trade caería a ~$0.06 neto.
- **No simula slippage** al ejecutar volúmenes grandes en Deribit.
- **No modela "fill probability"** si tu estrategia es postear limit orders
  en Kalshi en lugar de cruzar el spread.
- **No descuenta el coste del horizon mismatch** (Kalshi liquida en T_K,
  Deribit en T_D > T_K — la replicación no es perfecta).

---

## 6. Resumen visual del pipeline completo

```
SNAPSHOTTER (cada 60s)
      │
      ▼
data/snapshots/2026-05-09/172002.json.gz
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
  │      → 17,984 filas                      │
  │                                          │
  │   4. Métricas: Brier, log loss,          │
  │      calibración, PnL ejecutable         │
  └───────────────────┬──────────────────────┘
                      │
                      ▼
              data/reports/backtest.csv
                      │
                      ▼
        STREAMLIT APP (analytics interactivo)
```

---

## 7. Qué hemos descubierto (resultados)

### El bueno
- **El modelo q es estadísticamente bueno**: predice mucho mejor que los
  precios medios de Kalshi (Brier 46% mejor, log loss 2× mejor).
- **Está calibrado**: cuando dice 25% acierta ~25% de las veces.
- **Robusto al horizonte**: predice igual de bien a 5 min que a 60 min.

### El malo
- **Los spreads Kalshi son enormes** (mediana 1c en bins sin liquidez,
  hasta 30c en otros). El edge teórico se evapora al cruzar.
- **El 50% de bins no tienen bid** (nadie esperando comprar). No se
  pueden tradear aunque el modelo grite oportunidad.

### Lo que sigue siendo prometedor
- **Slice ultra-líquido (spread ≤ 3c, size ≥ 50, ~10% de bins)**:
  - Solo lado SELL YES (vender YES Kalshi + cubrirse con vertical Deribit).
  - 18 trades en 25h.
  - PnL +$1.67 bruto, avg +$0.093/trade, winrate 83%.
  - Distribuido en 10 events distintos (no un outlier).
  - Caveat: muestra pequeña, fees no descontados.

---

## 8. Estado actual (2026-05-10)

| Componente | Estado | Comentario |
|---|---|---|
| Snapshotter persistente | ✅ corriendo | 575+ snapshots, ~25h |
| Pipeline run_pipeline | ✅ | mid + bid/ask SVI fits |
| Backtest offline | ✅ | con grid de liquidez |
| Visualización findings | ✅ | findings.png estático |
| Dashboard Tkinter | ✅ vivo | snapshot actual (vigilancia) |
| Streamlit analytics | ✅ | histórico interactivo |
| Fees en backtest | ❌ pendiente | siguiente paso |
| Más datos (1 semana+) | ⏳ acumulando | clave antes de live |
| Paper trading | ⏳ | solo si edge persiste |

---

## 9. Qué hacer cuando vuelvas a este repo

```bash
cd /Users/carlosalonso/Cross_market_arbitrage

# Ver si el snapshotter sigue vivo:
ps -ef | grep snapshotter | grep -v grep

# Si no, arrancarlo:
bash start.sh

# Re-correr el backtest con los datos nuevos:
bash backtest.sh

# Abrir analytics:
bash analytics.sh

# Cuando empieces a operar, vista vivo:
bash dash.sh
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
