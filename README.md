<div align="center">

# 🚨 UPI Fraud Detection Using Machine Learning & Hybrid Security Engine

**Enterprise-Grade Real-Time Fraud Assessment Platform** — Random Forest Classifier · Unsupervised Anomaly Detection · Hybrid Rule Engine · Explainable AI (XAI) · Real-Time WebSockets · Geographic Threat Heatmap

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-success)
![Model Accuracy](https://img.shields.io/badge/accuracy-99.39%25-blueviolet)
![License](https://img.shields.io/badge/project-final--year%20B.Tech-orange)

</div>

---

## 🔴 Live Demo & Interfaces

### 🚀 [Transaction Risk Console — UPI Fraud Detection](https://eternalupi.netlify.app/index.html?api=https://upi-fraud-api-bvtv.onrender.com)

| 🖥️ Interface | 🔗 Link | 💡 Description |
|---|---|---|
| 🎛️ **Transaction Console** | [eternalupi.netlify.app](https://eternalupi.netlify.app/index.html?api=https://upi-fraud-api-bvtv.onrender.com) | Single-transaction risk analysis, XAI reason code breakdown, 1-click attack presets, hard rule banners. |
| 📡 **Real-Time Live Monitor** | [eternalupi.netlify.app/live.html](https://eternalupi.netlify.app/live.html?api=https://upi-fraud-api-bvtv.onrender.com) | Low-latency WebSocket stream monitor with live oscilloscope pulse animation & attack burst modes. |
| 🗺️ **Threat Heatmap & Analytics** | [eternalupi.netlify.app/analytics.html](https://eternalupi.netlify.app/analytics.html?api=https://upi-fraud-api-bvtv.onrender.com) | Leaflet.js interactive dark map plotting regional fraud density, 24h risk distribution, and 1-click CSV audit exporter. |
| 📘 **API Docs (Swagger)** | [upi-fraud-api-bvtv.onrender.com/docs](https://upi-fraud-api-bvtv.onrender.com/docs) | Interactive OpenAPI documentation for REST & WebSocket endpoints. |

> ⚠️ **Heads up:** The backend runs on Render's free tier, which sleeps after 15 minutes of inactivity. The first request may take 30-60 seconds to wake it up — if a demo link looks unresponsive at first, give it a moment ⏳ and try again.

---

## 📋 Table of Contents

- [✨ Highlights](#-highlights)
- [✅ Project Architecture & Key Phases](#-project-architecture--key-phases)
- [⚡ Quick Start](#-quick-start)
- [🔍 Phase 1: Explainable AI (XAI) Engine](#-phase-1-explainable-ai-xai-engine)
- [🛡️ Phase 2: Hybrid Business Rule Engine](#-phase-2-hybrid-business-rule-engine)
- [📡 Phase 3: Real-Time Streaming & 1-Click Attack Simulator](#-phase-3-real-time-streaming--1-click-attack-simulator)
- [🗺️ Phase 4: Threat Analytics & Geographic Heatmap](#-phase-4-threat-analytics--geographic-heatmap)
- [🧠 Model Performance & Real Data Validation](#-model-performance--real-data-validation)
- [📁 Project Structure](#-project-structure)
- [🧪 Running Tests](#-running-tests)
- [🐳 Docker & Deployment](#-docker--deployment)

---

## ✨ Highlights

- 🎯 **99.39% accuracy, 0.962 F1** with Random Forest — outperforming baseline research papers by up to 26 percentage points.
- 🕵️ **Unsupervised Anomaly Score** (Isolation Forest) correlating 0.65 with real fraud labels *without ever seeing labels during training*.
- 🔍 **Explainable AI (XAI)** decomposing every prediction into **Risk Drivers** (`▲ RISK FACTOR`) and **Trust Factors** (`▼ TRUST FACTOR`) with human-readable reason codes (`/predict/explain`).
- 🛡️ **Hybrid Security Rule Engine** enforcing hard deterministic checks (`Blacklist Entity`, `Velocity Cap`, `High Amount Limit`, `Triple Discrepancy`) with composite risk formula $\max(\text{ML}, \text{Rule\_Score})$.
- ⚡ **1-Click Fraud Attack Presets** (*SIM Swap*, *Micro-Probe*, *Device Hijack*, *Blacklist*, *Safe Transfer*) for instant live demonstration.
- 🗺️ **Geographic Threat Intelligence Map** plotting city-level fraud pulse markers across 16 Indian regions with 1-click CSV audit report exports.
- 🌍 **Validated on Real European Credit Card Data** — schema-agnostic pipeline achieving **99.84% accuracy** on independent Kaggle dataset (`mlg-ulb/creditcardfraud`).
- 🧪 **100% Passing Automated Tests** — 17 unit/integration test suites covering APIs, rules, streaming, and database logging.

---

## ✅ Project Architecture & Key Phases

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           UPI Fraud Detection Engine                            │
├──────────────────┬──────────────────┬─────────────────────┬─────────────────────┤
│ Phase 1: XAI     │ Phase 2: Rules   │ Phase 3: Simulator  │ Phase 4: Analytics  │
│ Feature          │ Deterministic    │ Async WebSockets    │ Geographic Heatmap  │
│ Attribution &    │ Hard Security    │ 1-Click Attack      │ 24h Risk Histogram  │
│ Reason Codes     │ Rule Engine      │ Scenario Streams    │ CSV Audit Exporter  │
└──────────────────┴──────────────────┴─────────────────────┴─────────────────────┘
```

- [x] **Phase 1: Explainable AI (XAI)** — Feature contribution breakdown and human-readable reason codes (`app/explain.py`, `POST /predict/explain`).
- [x] **Phase 2: Hybrid Rule Engine & Composite Risk** — Policy evaluation (`Blacklist`, `Velocity Cap`, `Single Amount Cap`, `Triple Discrepancy`) with composite scoring $\max(\text{ML}, \text{Rule})$ (`app/rules.py`, `GET /rules`, `POST /rules/update`).
- [x] **Phase 3: Real-Time Stream & Attack Simulator** — Async WebSocket feed (`ws://.../ws/live?speed=1.0&scenario=sim_swap`) with live oscilloscope pulse animation & 1-click attack toolbar (`app/stream.py`, `frontend/live.html`).
- [x] **Phase 4: Visual Analytics & Threat Heatmap** — Leaflet.js dark map displaying city-level fraud pulse markers across 16 Indian regions, 24-hour distribution charts, and CSV audit report generator (`app/db.py`, `frontend/analytics.html`, `/analytics/export`).

---

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/shashwat-eternal/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection

# 2. Install dependencies (virtual environment recommended)
pip install -r requirements.txt

# 3. Launch FastAPI backend server
python -m uvicorn app.main:app --reload
```

Then open any of the HTML dashboards directly in your browser — zero build step, zero npm install! 
- Console: `frontend/index.html`
- Live Stream: `frontend/live.html`
- Analytics Map: `frontend/analytics.html`

---

## 🔍 Phase 1: Explainable AI (XAI) Engine

**Endpoint**: `POST /predict/explain`

Computes feature-level contribution attributions and outputs plain-English reason codes:
- **Risk Drivers (`▲ RISK FACTOR`)**: Features increasing risk probability (e.g. *First-time transfer to unverified beneficiary*, *Geographic location mismatch detected*).
- **Trust Factors (`▼ TRUST FACTOR`)**: Features establishing legitimate behavior (e.g. *Standard transaction velocity pattern*).

```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.9842,
  "composite_risk_score": 0.9842,
  "explanation": {
    "risk_drivers": [
      { "reason_code": "First-time transfer to an unverified beneficiary", "impact_score": 0.85 },
      { "reason_code": "Geographic location mismatch detected (IP vs Home location)", "impact_score": 0.82 }
    ]
  }
}
```

---

## 🛡️ Phase 2: Hybrid Business Rule Engine

**Module**: `app/rules.py` | **Endpoints**: `GET /rules`, `POST /rules/update`

Combines machine learning predictions with hard security policy rules:
1. **`RULE_001_BLACKLIST`**: Immediately blocks blacklisted banks or entity VPAs.
2. **`RULE_002_VELOCITY_BREACH`**: Flags when 24h frequency exceeds threshold.
3. **`RULE_003_AMOUNT_CAP`**: Flags single transactions exceeding limit (default: ₹50,000).
4. **`RULE_004_TRIPLE_FLAG`**: Triggers when beneficiary, location, and device change occur together.

$$\text{Composite Risk Score} = \max(\text{ML\_Probability}, \text{Rule\_Risk\_Score})$$

---

## 📡 Phase 3: Real-Time Streaming & 1-Click Attack Simulator

**WebSocket**: `ws://127.0.0.1:8000/ws/live?speed=1.0&scenario=sim_swap`

Features an infinite transaction stream scored in real time with continuous oscilloscope pulse wave animation (`frontend/live.html`) and 1-click synthetic attack scenario presets:
- ⚡ **SIM Swap Attack**: High amount (₹48,500), 2 AM execution, device + location mismatch.
- 🔄 **Micro-Probe**: ₹2.50 micro-transfer with high 24h velocity.
- 🎭 **Device Hijack**: Multi-factor discrepancy breach.
- 🚫 **Blacklist Entity**: Known fraud entity VPA.
- 🛡️ **Safe Transfer**: Standard daytime transfer.

---

## 🗺️ Phase 4: Threat Analytics & Geographic Heatmap

**Dashboard**: `frontend/analytics.html` | **Endpoints**: `/analytics/locations`, `/analytics/hourly`, `/analytics/export`

- **Interactive Leaflet.js Map**: Plots regional fraud density with glowing pulse markers across 16 Indian cities (Lucknow, Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata, Pune, Jaipur, Patna, Bhopal, Kanpur, Rural-UP, Rural-Bihar, Rural-MP, Rural-Rajasthan).
- **24-Hour Fraud Histogram**: Visualizes peak fraud hours and transaction frequency.
- **CSV Audit Exporter**: 1-click download of logged transaction history as `upi_fraud_audit_report.csv`.

---

## 🧠 Model Performance & Real Data Validation

### 1. Synthetic UPI Dataset (150,000 Transactions)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 🌳 Decision Tree | 99.14% | 93.49% | 95.85% | 0.947 | 0.979 |
| 🌲 **Random Forest** 🏆 | **99.39%** | **95.08%** | **97.32%** | **0.962** | **0.999** |
| 📍 KNN | 97.88% | 79.47% | 98.91% | 0.881 | 0.996 |

### 2. Independent Real Credit Card Dataset (ULB European Cardholders)
Validated via schema-agnostic generic pipeline (`src/data/generic_pipeline.py`):
- **Accuracy**: **99.84%**
- **F1 Score**: **0.916**
- **ROC-AUC**: **0.981**

---

## 📁 Project Structure

```
├── app/            ⚙️  FastAPI backend (XAI, Rule Engine, WebSockets, DB Analytics)
│   ├── main.py     ├── REST & WebSocket routing
│   ├── predict.py  ├── ML inference & pipeline evaluation
│   ├── explain.py  ├── XAI reason code generator (Phase 1)
│   ├── rules.py    ├── Hybrid rule engine & config (Phase 2)
│   ├── stream.py   ├── WebSocket stream & attack generator (Phase 3)
│   └── db.py       └── SQLite logger & analytics (Phase 4)
├── data/           📊  Raw, processed, external datasets & predictions.db
├── frontend/       🎛️  Dashboards (index.html, live.html, analytics.html)
├── models/         🧠  Trained Random Forest, preprocessor & Isolation Forest pkls
├── notebooks/      📓  Step-by-step analysis notebooks (Day 1 - Day 8)
├── reports/        📄  Comprehensive documentation, reports & interview prep
├── src/            🔧  Reusable Python packages (data, features, generic pipeline)
└── tests/          🧪  Automated pytest suite (17 test cases, 100% passing)
```

---

## 🧪 Running Tests

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

✅ **17/17 passed (100%)** — covering APIs, XAI reason codes, Rule Engine evaluations, WebSockets, and SQLite analytics export.

---

## 🐳 Docker & Deployment

```bash
# Build container image
docker build -t upi-fraud-api .

# Run container locally
docker run -p 8000:8000 upi-fraud-api
```

Live Deployed Services:
- **Backend**: Render Docker Web Service (`render.yaml`)
- **Frontend**: Netlify Static Site

---

<div align="center">

**Built by [Shashwat](https://github.com/shashwat-eternal)** · B.Tech CSE, Final Year · Babu Banarasi Das Institute of Technology and Management (AKTU)

🚨 *If it flags your transaction as fraud, it's probably just really cautious.* 🚨

</div>