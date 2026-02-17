# The Earnings Hunter

An AI-powered **earnings analysis platform** that combines hard financial data, CEO tone, news sentiment, insider activity, and ML predictions into a single **composite confidence score**.

The app features a **Financial Expert Agent** (GPT-4o-mini) that automatically scores each data component, weighted into a 5-factor composite displayed with an institutional-grade dashboard.

---

## Key Features

- **5-Component Composite Scoring**
  - Financial (25%) + CEO Tone (20%) + News (15%) + Insider (10%) + ML Model (30%)
  - AI-powered scoring with reasoning for each component
  - Displayed as animated confidence ring + score breakdown bars
- **Financial Expert Agent**
  - GPT-4o-mini scores 4 data categories (0-100) with reasoning
  - Runs automatically on every search (~$0.01 per analysis)
  - Receives raw data: financials, transcript, news headlines, insider trades
- **Modern React Dashboard**
  - React + TypeScript + Vite frontend with institutional-grade UI design
  - 5-axis Analysis Radar chart + animated confidence ring
  - Score Breakdown with color-coded progress bars and hover reasoning
  - Responsive layout with Tailwind CSS
- **FastAPI Backend**
  - Python-powered REST API wrapping ML models and data pipelines
  - Composite breakdown computation from expert scores + ML consensus
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
│  - Confidence Ring + Score Breakdown (5 components)         │
│  - 5-axis Analysis Radar / Golden Triangle fallback         │
│  - MarketStage (PriceChart, TimeframePills)                 │
│  - FinancialIntel (Financials, EarningsCall, News, Insider) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  - /api/analyze/{symbol}    → Full analysis + composite     │
│  - /api/quote/{symbol}      → Real-time quote               │
│  - /api/historical/{symbol} → Price history for charts      │
│  - /api/deep-analysis/{symbol} → CrewAI agents (optional)   │
│  - Composite breakdown: 5-component weighted score          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              ML Pipeline + Expert Agent                      │
│  - EarningsOrchestrator → Coordinates all data fetching     │
│  - FeaturePipeline → Extracts 48 features                   │
│  - EarningsPredictor → 5 ML models + consensus voting       │
│  - FinancialExpertScorer → GPT-4o-mini (scores 4 components)│
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
| **Expert Agent** | OpenAI `gpt-4o-mini` (direct API, automatic scoring) |
| **Deep Analysis** | CrewAI + OpenAI `gpt-4o-mini` (optional) |
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
│       ├── financial_scorer.py # GPT-4o-mini expert agent (scores 4 components)
│       ├── crew.py           # CrewAI 3-agent system (optional deep analysis)
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

## Composite Scoring System

The confidence score is computed from **5 weighted components**:

| Component | Weight | Source | Scorer |
|-----------|--------|--------|--------|
| **Financial** | 25% | EPS/revenue surprises, margins, price targets | GPT-4o-mini Expert Agent |
| **CEO Tone** | 20% | Earnings call transcript excerpt | GPT-4o-mini Expert Agent |
| **News** | 15% | Stock news headlines + sentiment | GPT-4o-mini Expert Agent |
| **Insider** | 10% | Insider buy/sell transactions | GPT-4o-mini Expert Agent |
| **ML Model** | 30% | (confidence × 0.6 + agreement × 0.4) × 100 | Algorithmic |

Each component is scored **0-100** with reasoning. The weighted sum produces the final composite confidence displayed in the ring.

**Score color coding:**
- Green: >= 65 (bullish signal)
- Amber: 35-64 (neutral/mixed)
- Red: < 35 (bearish signal)

---

## Golden Triangle Framework (Feature Extraction)

Used for ML **feature extraction** (48 features fed to models):

| Vector | Weight | Source | Features |
|--------|--------|--------|----------|
| **Hard Data** | 40% | FMP API | EPS/revenue surprises, margins, analyst consensus, insider sentiment |
| **CEO Tone** | 35% | Transcript NLP | Confidence, sentiment, guidance, uncertainty (VADER + TextBlob) |
| **Social** | 25% | News NLP | News sentiment, bullish/bearish ratio |

**Note:** These weights are for feature extraction only. The final display score uses the 5-component composite weights above.

---

## CrewAI Deep Analysis (Optional)

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

### 2026-02-17: Composite Scoring System + Financial Expert Agent
- Added GPT-4o-mini Financial Expert Agent (`src/agents/financial_scorer.py`)
- Replaced simple confidence calculation with 5-component weighted composite score
- Agent automatically scores Financial, CEO Tone, News, and Insider data (0-100 with reasoning)
- ML Model score computed algorithmically from consensus data
- Added Score Breakdown UI with color-coded progress bars and hover reasoning
- Upgraded radar chart from 3-axis Golden Triangle to 5-axis Analysis Radar
- Added `CompositeBreakdown`, `MLConsensus` TypeScript types
- Added `expert_scores`, `ml_consensus` fields to orchestrator
- Added `_compute_composite_breakdown()` to API layer

### 2026-02-13: React Frontend + FastAPI Backend
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
