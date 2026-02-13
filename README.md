# The Earnings Hunter

An AI-powered **earnings analysis platform** that combines hard financial data, CEO tone, and news sentiment into a single, modern dashboard.
The app implements a **"Golden Triangle"** framework (40% Financials + 35% CEO Tone + 25% Social/News) plus optional multi-agent deep analysis via CrewAI.

---

## Key Features

- **Golden Triangle Scoring**
  - Blends earnings surprises, margins, insider activity, CEO tone, and news sentiment into a single 0-10 score.
- **Modern React Dashboard**
  - React + TypeScript + Vite frontend with institutional-grade UI design
  - Real-time data visualization with Recharts
  - Responsive layout with Tailwind CSS
- **FastAPI Backend**
  - Python-powered REST API wrapping ML models and data pipelines
  - CORS-enabled for frontend integration
- **ML Earnings Outlook**
  - Multiple trained models (LightGBM, XGBoost, Random Forest, Logistic Regression, Neural Net)
  - Consensus prediction: **Growth** / **Stagnation** / **Risk** + confidence score
  - Trained on 1,050+ real earnings quarters from FMP data
- **CEO Tone & NLP**
  - Analyzes earnings call transcripts using VADER + TextBlob NLP pipeline
  - Extracts confidence, sentiment, guidance signals
- **News-Driven Social Sentiment**
  - Uses FMP `news/stock-latest` + NLP; aggregates bullish/bearish ratios
- **Optional CrewAI Deep Analysis**
  - Three agents (Scout, Social Listener, Fusion) generate research-style narrative reports
- **Production-Ready**
  - Dockerfile + `docker-compose.yml`, scripts for data collection and model training

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                    │
│  - LegalDisclaimer, CommandBar, TickerStrip                 │
│  - IntelligenceHub (ConfidenceRing, RadarChart, AIAnalysis) │
│  - MarketStage (PriceChart, TimeframePills)                 │
│  - FinancialIntel (Financials, EarningsCall, News, Insider) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  - /api/analyze/{symbol}    → Full analysis + ML prediction │
│  - /api/quote/{symbol}      → Real-time quote               │
│  - /api/historical/{symbol} → Price history for charts      │
│  - /api/deep-analysis/{symbol} → CrewAI agents (optional)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python ML Pipeline                        │
│  - EarningsOrchestrator → Coordinates all data fetching     │
│  - FeaturePipeline → Extracts 48 features                   │
│  - EarningsPredictor → 5 ML models + consensus voting       │
│  - TranscriptAnalyzer → CEO tone NLP                        │
│  - SentimentFeatureExtractor → News sentiment NLP           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FMP API (Data Source)                     │
│  - /stable/quote, earnings-surprises, income-statement      │
│  - /stable/earnings-call-transcript, insider-trading        │
│  - /stable/news/stock-latest, historical-price-eod          │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Recharts, Framer Motion, Radix UI |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML** | scikit-learn, XGBoost, LightGBM, custom trainer & predictor |
| **NLP** | VADER, TextBlob |
| **Agents** | CrewAI + OpenAI (`gpt-4o-mini`) |
| **Data** | FMP `/stable/` API |
| **Infra** | Docker, Docker Compose |

---

## Project Structure

```
earnings_hunter/
├── api/                       # FastAPI backend
│   ├── main.py               # FastAPI app entry point
│   └── routers/
│       ├── analysis.py       # /analyze/{symbol} endpoint
│       ├── quote.py          # /quote/{symbol} endpoint
│       └── historical.py     # /historical/{symbol} endpoint
│
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── LegalDisclaimer.tsx
│   │   │   ├── CommandBar.tsx
│   │   │   ├── TickerStrip.tsx
│   │   │   ├── IntelligenceHub/
│   │   │   ├── MarketStage/
│   │   │   └── FinancialIntel/
│   │   ├── services/api.ts   # API client
│   │   ├── types/index.ts    # TypeScript interfaces
│   │   └── App.tsx           # Main app component
│   ├── tailwind.config.js    # Design system colors
│   └── package.json
│
├── src/                       # Python ML/Data pipeline
│   ├── data_ingestion/
│   │   ├── fmp_client.py     # FMP API client
│   │   └── validators.py     # Pydantic models
│   ├── feature_engineering/
│   │   ├── financial_features.py
│   │   ├── transcript_analyzer.py
│   │   ├── sentiment_features.py
│   │   ├── social_features.py
│   │   └── feature_pipeline.py
│   ├── ml/
│   │   ├── model_comparison.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   └── agents/
│       ├── orchestrator.py   # Main analysis coordinator
│       ├── crew.py           # CrewAI 3-agent system
│       └── tools/fmp_tools.py
│
├── config/settings.py         # Pydantic settings
├── scripts/                   # CLI utilities
│   ├── collect_training_data.py
│   ├── train_model.py
│   └── run_analysis.py
├── data/
│   ├── models/               # Trained ML models (5 models)
│   └── training/             # Training datasets
│
├── app/                       # Legacy Streamlit (deprecated)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── CLAUDE.md                  # Project memory & decisions
└── README.md                  # This file
```

---

## Requirements

- **Python**: 3.11+
- **Node.js**: 18+ (for frontend)
- **APIs**:
  - FMP **Ultimate** (or compatible) API key
  - OpenAI API key (for CrewAI deep analysis)

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required
FMP_API_KEY=your_fmp_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional
LOG_LEVEL=INFO
CACHE_EXPIRY_HOURS=24
```

---

## Quick Start

### Option 1: Run Locally (Development)

```bash
# 1. Clone and setup Python environment
cd "Earnings Hunter Fnal"
conda activate earnings_hunter  # or your preferred env
pip install -r requirements.txt

# 2. Setup frontend
cd frontend
npm install
cd ..

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start Backend (Terminal 1)
python -m uvicorn api.main:app --reload --port 8000

# 5. Start Frontend (Terminal 2)
cd frontend
npm run dev

# 6. Open browser
# http://localhost:5173
```

### Option 2: Docker

```bash
docker-compose up --build
# App available at http://localhost:5173
# API available at http://localhost:8000
```

---

## ML Models

The system uses **5 trained ML models** with consensus voting:

| Model | CV Accuracy | Notes |
|-------|-------------|-------|
| Neural Network (MLP) | 53.9% | **Best model** |
| LightGBM | 51.4% | Fast, good for production |
| Random Forest | 51.0% | Robust baseline |
| XGBoost | 50.0% | Gradient boosting |
| Logistic Regression | 39.9% | Interpretable |

**Training Data**: 1,050 real earnings quarters from FMP (2021-2026)

**Labels** (based on 5-day post-earnings price change):
- **Growth**: Price increased > 5%
- **Risk**: Price decreased > 5%
- **Stagnation**: Price change between -5% and +5%

### Retrain Models

```bash
python scripts/train_model.py \
  --data data/training/training_data_20260206_124508.csv \
  --output-dir data/models
```

---

## Golden Triangle Framework

| Vector | Weight | Source | Features |
|--------|--------|--------|----------|
| **Hard Data** | 40% | FMP API | EPS/revenue surprises, margins, analyst consensus, insider sentiment |
| **CEO Tone** | 35% | Transcript NLP | Confidence, sentiment, guidance, uncertainty (VADER + TextBlob) |
| **Social** | 25% | News NLP | News sentiment, bullish/bearish ratio |

**48 total features** are extracted and fed to the ML models.

---

## CrewAI Deep Analysis

When enabled, runs a **three-agent system**:

| Agent | Role | Tools |
|-------|------|-------|
| **Scout Agent** | Financial + CEO tone analysis | FMP earnings, transcripts, insider data |
| **Social Listener** | News sentiment analysis | FMP stock news |
| **Fusion Agent** | Synthesizes all data | Receives other agents' output |

**Cost**: ~$0.01-0.02 per deep analysis
**Time**: 30-60 seconds

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/{symbol}` | GET | Full analysis with Golden Triangle + ML prediction |
| `/api/quote/{symbol}` | GET | Real-time stock quote |
| `/api/historical/{symbol}` | GET | Historical prices for charts |
| `/api/deep-analysis/{symbol}` | POST | CrewAI agent analysis |

Example:
```bash
curl http://localhost:8000/api/analyze/NVDA
```

---

## Changelog

### 2026-02-12: React Frontend + FastAPI Backend
- Replaced Streamlit with React + TypeScript + Vite
- Created FastAPI backend wrapping existing ML pipeline
- Added institutional-grade UI design
- Fixed ML model version mismatch (retrained models)
- Added transcript, news, insider data to API responses

### 2026-02-09: Single-Page Streamlit App
- Converted multi-page to onepager
- Fixed Golden Triangle display
- Added CrewAI deep analysis integration

### 2026-02-05: Initial Release
- FMP data integration
- ML model training pipeline
- NLP sentiment analysis

---

## Disclaimer

This project is for **educational and research purposes only** and **does not constitute financial advice**.
Predictions, scores, and visualizations are experimental and should **not** be used as the sole basis for any investment decision.
Always do your own research and consult a licensed financial professional.

---

## License

MIT License - see LICENSE file for details.
