# Crypto SMC Candidate Scanner

A standalone personal crypto perpetual market scanner built for **manual trading candidate discovery**.

The scanner reduces the large Binance USDT-M perpetual futures universe (300+ markets) to approximately **5–15 high-quality candidates** that deserve manual chart and orderflow inspection on TradingView and ATAS.

> **Note:** This is **NOT** an automated trading bot. It does not place trades or make autonomous trading decisions. Its sole objective is market sorting and candidate discovery.

---

## ⚡ High-Level Workflow

```text
Crypto Perpetual Universe (Binance USDT-M)
                    ↓
Liquidity Filter (24H Volume >= $50M)
                    ↓
RVOL Filter (1H Completed RVOL >= 1.5x)
                    ↓
Volatility Filter (|24H Price Change| >= 3%)
                    ↓
Open Interest Filter (1H OI Change >= +5%)
                    ↓
Price/OI Classification
                    ↓
SMC Context (4H/1H/15M Structure, Swings, BOS, MSS, Displacement, OB, FVG, Sweeps, HTF Location)
                    ↓
Candidate Ranking (Deterministic 100-Point Score)
                    ↓
TOP 5–15 Candidates
                    ↓
Manual Chart & ATAS Orderflow Inspection
```

---

## 🚀 Quick Start

### 1. Requirements

* Python 3.11+
* Dependencies: CCXT, Streamlit, Pandas, NumPy, Requests, Python-dotenv, Google-GenAI, Pytest

### 2. Setup

Clone or open the directory:
```bash
cd "c:/Users/DELL/Desktop/Data Engineering Projects/Trade Finder"
```

Install dependencies:
```bash
pip install -r requirements.txt
```

*(Optional)* Configure your environment in `.env`:
```bash
cp .env.example .env
```
Provide your optional `GEMINI_API_KEY` if you want AI-assisted setup explanations. (The scanner runs 100% deterministically without an API key).

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`.

---

## 🎯 Core Features

### 1. Market Universe
* Scans active Binance USDT-settled perpetual contracts.
* Excludes inactive, delisted, and delivery/quarterly futures.
* Extensible architecture for adding Bybit in the future.

### 2. Quantitative Screening Filters
* **Liquidity Filter:** Default 24H quote volume $\ge$ $50,000,000 (configurable).
* **Relative Volume (RVOL):** 
  $$\text{RVOL} = \frac{\text{Current completed 1H volume}}{\text{Median volume of previous 20 completed 1H candles}}$$
  Only completed hourly candles are used for confirmed RVOL. Default threshold $\ge$ 1.5x.
* **Volatility Filter:** Absolute 24H price change $\ge$ 3.0%.
* **Open Interest (OI):** 1H and 4H timestamped OI percentage changes. Default 1H OI change $\ge$ +5.0%.
* **Price / OI Discovery Matrix:**
  * Price falling + OI rising: `OI BUILD + PRICE WEAKNESS`
  * Price rising + OI rising: `OI BUILD + PRICE STRENGTH`
  * Price falling + OI falling: `OI CONTRACTION + PRICE WEAKNESS`
  * Price rising + OI falling: `OI CONTRACTION + PRICE STRENGTH`

### 3. Smart Money Concepts (SMC) Engine
Implemented strictly with pure Python, Pandas, and NumPy (zero external SMC library dependencies):
* **Wilder's RMA ATR:** Period = 14.
* **Chronological Swings:** Lookback = 5 on left and right, with zero lookahead bias and deterministic tie-breaking.
* **Break of Structure (BOS):** Requires candle body close beyond confirmed swing levels (wicks do not trigger BOS).
* **Displacement:** Consecutive same-direction candle bodies with total displacement $\ge$ 1.5x ATR within a 3-bar window.
* **Market Structure Shift (MSS):** Opposite-direction break of confirmed structure accompanied by qualifying displacement.
* **Order Blocks (OB):** Last opposing candle before displacement move. Includes doji fallback (body < 0.1x ATR) and lifecycle tracking (`ACTIVE`, `MITIGATED`, `INVALIDATED`).
* **Fair Value Gaps (FVG):** 3-candle imbalance structure with min gap $\ge$ 0.25x ATR and state tracking (`OPEN`, `PARTIAL`, `FILLED`).
* **Liquidity Pools & Sweeps:** Equal highs/lows (EQH/EQL), swing extremes, PDH/PDL, and sweep detection (wicks beyond pool followed by close back inside).
* **HTF Location:** Identifies whether current price is `NEAR DEMAND`, `NEAR SUPPLY`, `NEAR LIQUIDITY`, `MID-RANGE`, or `NO SIGNIFICANT LOCATION`.

### 4. Deterministic 100-Point Candidate Score
Score represents **how interesting a candidate is according to strategy rules**, not win probability:
| Component | Default Weight |
|---|---|
| Liquidity | 10 pts |
| RVOL | 15 pts |
| Open Interest | 15 pts |
| Volatility | 10 pts |
| HTF Location | 15 pts |
| Liquidity Sweep | 10 pts |
| Displacement | 10 pts |
| BOS / MSS | 10 pts |
| OB / FVG Confluence | 5 pts |
| **Total** | **100 pts** |

### 5. Candidate Stages & Actionable Context
Candidates are categorized into:
* `WATCH`
* `SETUP_FORMING`
* `CONFIRMED`
* `NO_TRADE`

Every candidate card and inspector panel answers two essential trading questions:
1. **Why is this coin interesting?** (Highlights high RVOL, OI expansion, HTF demand, swept liquidity).
2. **What am I waiting for before considering a trade?** (e.g. `Waiting for: Bullish liquidity sweep + MSS on 1H/15M`).

### 6. Reference Trade Plan
When qualifying SMC structure is present (Active OB in setup direction), the dashboard generates a `REFERENCE TRADE PLAN`:
* Entry Zone
* Stop Loss ($OB.low - 1.0 \times ATR$ for long, $OB.high + 1.0 \times ATR$ for short)
* 1R, 2R, 3R targets and Risk/Reward ratio.

### 7. ATAS Orderflow Confirmation Checklist
Manual orderflow review checklist in the Streamlit inspector:
* Footprint reviewed
* Delta supports setup
* Absorption observed
* POC behaviour supports setup
* Imbalance supports setup
* 1M/5M MSS confirmed
* Status: `NOT CHECKED` / `SUPPORTS` / `DOES NOT SUPPORT`

### 8. Optional Gemini AI Setup Explainer
Click **"🤖 Explain Setup"** in the candidate inspector to synthesize deterministic evidence with Google Gemini into structured assessments without hallucinating missing metrics.

---

## 🧪 Testing

Run the test suite with pytest:
```bash
python -m pytest tests/ -v
```

All 13 test suites validate:
* Wilder RMA ATR calculation
* RVOL calculation & volume categorization
* 1H & 4H OI metrics & classification matrix
* Chronological swing detection (zero lookahead)
* Break of Structure (BOS) closing requirements
* Displacement detection
* Order Block detection & lifecycle state tracking
* Fair Value Gap (FVG) detection & fill tracking
* Liquidity sweep detection & HTF location classification
* 100-point candidate scoring breakdown

---

## 📁 Project Structure

```text
Trade Finder/
│
├── app.py                     # Streamlit web dashboard
├── config.py                  # Scanner configurations and weights
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── .env.example               # Environment variables template
│
├── data/
│   ├── cache.py               # TTL in-memory cache
│   └── exchange.py            # Binance perpetuals client & demo data
│
├── indicators/
│   ├── atr.py                 # Wilder RMA ATR
│   ├── volume.py              # 1H completed RVOL
│   ├── oi.py                  # Open Interest changes & classification
│   └── volatility.py          # 24H volatility filter
│
├── smc/
│   ├── swings.py              # Confirmed swing highs & lows
│   ├── displacement.py        # Displacement detector
│   ├── structure.py           # BOS, MSS, and bias analysis
│   ├── order_blocks.py        # Order block lifecycle manager
│   ├── fvg.py                 # 3-candle Fair Value Gaps
│   └── liquidity.py           # Liquidity pools, sweeps & HTF location
│
├── scanner/
│   ├── filters.py             # Quantitative screening filters
│   ├── scoring.py             # 100-point scoring engine
│   ├── candidates.py          # Candidate builder & reference plan
│   └── pipeline.py            # End-to-end scanner orchestrator
│
├── ai/
│   └── gemini.py              # Optional Gemini AI setup explainer
│
└── tests/                     # Unit test suite (13 passing tests)
```
