# CLAUDE.md - Project Memory & Progress Tracker

> **IMPORTANT**: This file is the project's memory. Update it after EVERY change!
> Last Updated: 2026-02-13 (React Frontend + FastAPI Backend)

## Project Overview
**Name:** The Earnings Hunter
**Goal:** Multi-agent SaaS for earnings analysis using Golden Triangle (40% Financial + 35% CEO Tone + 25% Social)
**Status:** PRODUCTION - React + FastAPI Architecture
**Data Source:** FMP API ONLY (no Reddit/PRAW)

---

## MAJOR ARCHITECTURE CHANGE: Streamlit → React + FastAPI

### Timeline:
1. **2026-02-05**: Initial Streamlit multi-page app
2. **2026-02-09**: Converted to single-page Streamlit app
3. **2026-02-13**: **Complete rewrite to React + FastAPI architecture**

### Current Architecture (2026-02-13):

```
┌─────────────────────────────────────────┐
│      React Frontend (Vite + TS)         │
│  - Institutional Cybernetics Design     │
│  - Real-time data visualization         │
│  - Recharts for charts                  │
│  - Tailwind CSS + Framer Motion         │
└─────────────────────────────────────────┘
                  ↓ HTTP/REST
┌─────────────────────────────────────────┐
│         FastAPI Backend (Python)        │
│  - /api/analyze/{symbol}                │
│  - /api/quote/{symbol}                  │
│  - /api/historical/{symbol}             │
│  - /api/deep-analysis/{symbol}          │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│    Existing ML Pipeline (Preserved)     │
│  - EarningsOrchestrator                 │
│  - FeaturePipeline (48 features)        │
│  - 5 ML Models + Consensus              │
│  - TranscriptAnalyzer (NLP)             │
│  - SentimentFeatureExtractor (NLP)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│          FMP API (/stable/)             │
└─────────────────────────────────────────┘
```

---

## File Creation Progress

### Phase 0-7: Original Python/Streamlit (COMPLETED)
All original files completed (see previous version for details)

### Phase 8: React Frontend (NEW - 2026-02-13)

| # | File | Status | Notes |
|---|------|--------|-------|
| 39 | frontend/package.json | ✅ Done | React 18, TypeScript, Vite, Tailwind |
| 40 | frontend/tailwind.config.js | ✅ Done | Institutional Cybernetics color palette |
| 41 | frontend/vite.config.ts | ✅ Done | Proxy to backend on :8000 |
| 42 | frontend/src/types/index.ts | ✅ Done | TypeScript interfaces matching API |
| 43 | frontend/src/services/api.ts | ✅ Done | API client with error handling |
| 44 | frontend/src/components/LegalDisclaimer.tsx | ✅ Done | Radix UI Dialog |
| 45 | frontend/src/components/CommandBar.tsx | ✅ Done | Ticker search with icon |
| 46 | frontend/src/components/TickerStrip.tsx | ✅ Done | Slot machine animation |
| 47 | frontend/src/components/SlotNumber.tsx | ✅ Done | Rolling number animation |
| 48 | frontend/src/components/IntelligenceHub/ConfidenceRing.tsx | ✅ Done | SVG circular progress |
| 49 | frontend/src/components/IntelligenceHub/GoldenTriangleRadar.tsx | ✅ Done | Recharts RadarChart |
| 50 | frontend/src/components/IntelligenceHub/AIAnalysis.tsx | ✅ Done | Deep analysis panel |
| 51 | frontend/src/components/IntelligenceHub/index.tsx | ✅ Done | Main hub component |
| 52 | frontend/src/components/MarketStage/PriceChart.tsx | ✅ Done | Recharts AreaChart |
| 53 | frontend/src/components/MarketStage/TimeframePills.tsx | ✅ Done | Tab selector |
| 54 | frontend/src/components/MarketStage/index.tsx | ✅ Done | Market stage container |
| 55 | frontend/src/components/FinancialIntel/FinancialsCard.tsx | ✅ Done | EPS/Revenue metrics |
| 56 | frontend/src/components/FinancialIntel/EarningsCallCard.tsx | ✅ Done | CEO tone display |
| 57 | frontend/src/components/FinancialIntel/NewsCard.tsx | ✅ Done | News sentiment |
| 58 | frontend/src/components/FinancialIntel/InsiderActivityCard.tsx | ✅ Done | Insider trades table |
| 59 | frontend/src/components/FinancialIntel/index.tsx | ✅ Done | Financial intel grid |
| 60 | frontend/src/components/FooterDisclaimer.tsx | ✅ Done | Footer component |
| 61 | frontend/src/App.tsx | ✅ Done | Main app with state management |
| 62 | frontend/index.html | ✅ Done | HTML entry point |

### Phase 9: FastAPI Backend (NEW - 2026-02-13)

| # | File | Status | Notes |
|---|------|--------|-------|
| 63 | api/main.py | ✅ Done | FastAPI app with CORS |
| 64 | api/routers/analysis.py | ✅ Done | Main analysis endpoint |
| 65 | api/routers/quote.py | ✅ Done | Real-time quote endpoint |
| 66 | api/routers/historical.py | ✅ Done | Historical prices endpoint |

---

## Completed Actions Log
<!-- Add new entries at the TOP -->

| Timestamp | Action | Files Changed | Notes |
|-----------|--------|---------------|-------|
| 2026-02-13 | **RETRAINED ML MODELS** | data/models/*.pkl | Fixed numpy/sklearn version mismatch |
| 2026-02-13 | **FIXED DATA FLOW** | orchestrator.py, analysis.py | Added transcript_content, news_articles, insider_transactions to API |
| 2026-02-13 | **CREATED FASTAPI BACKEND** | api/main.py, api/routers/*.py | REST API wrapping ML pipeline |
| 2026-02-13 | **CREATED REACT FRONTEND** | frontend/src/**/*.tsx | Complete UI rewrite with TypeScript |
| 2026-02-13 | **ARCHITECTURE REDESIGN** | Multiple files | Streamlit → React + FastAPI |
| 2026-02-09 | **CONVERTED to ONEPAGER** | app/main.py (rewritten), app/pages/* (deleted) | Single-page Streamlit app |
| 2026-02-09 | **FIXED** API Endpoints + NLP Sentiment | fmp_client.py, orchestrator.py | Stock news + VADER/TextBlob |
| 2026-02-09 | **ADDED** CrewAI Integration | crew.py | 3 agents, GPT-4o-mini |
| 2026-02-05 | **MIGRATED** Reddit → FMP Social | Multiple files | Deleted reddit_client, rate_limiter |
| 2026-02-05 | Created Docker files | Dockerfile, docker-compose.yml | Deployment ready |
| 2026-02-05 | Created scripts | scripts/*.py | Data collection, training |
| 2026-02-05 | Created agents module | src/agents/*.py | CrewAI crew |
| 2026-02-05 | Created ML module | src/ml/*.py | Multi-model |
| 2026-02-04 | Created feature engineering | src/feature_engineering/*.py | Golden Triangle |
| 2026-02-04 | Created data ingestion | src/data_ingestion/*.py | FMP client |

---

## React Frontend Architecture

### Design System
Based on "Institutional Cybernetics" aesthetic:

**Color Palette:**
```javascript
colors: {
  cyber: {
    bg: '#0a0e1a',        // Deep space background
    surface: '#111827',   // Card background
    border: '#1f2937',    // Borders
    accent: '#3b82f6',    // Primary blue
    success: '#10b981',   // Green (Growth)
    warning: '#f59e0b',   // Amber (Stagnation)
    danger: '#ef4444',    // Red (Risk)
    muted: '#6b7280',     // Gray text
  }
}
```

### Component Structure
```
App.tsx (Main state management)
├── LegalDisclaimer (Radix Dialog)
├── CommandBar (Search input)
├── TickerStrip (Animated ticker + slot numbers)
├── IntelligenceHub
│   ├── ConfidenceRing (SVG circular progress)
│   ├── GoldenTriangleRadar (Recharts radar)
│   └── AIAnalysis (Deep analysis panel)
├── MarketStage
│   ├── TimeframePills (Tab selector)
│   └── PriceChart (Recharts area chart)
├── FinancialIntel
│   ├── FinancialsCard (EPS/Revenue)
│   ├── EarningsCallCard (CEO tone)
│   ├── NewsCard (Sentiment)
│   └── InsiderActivityCard (Trades table)
└── FooterDisclaimer
```

### Key Features
- **Type Safety**: Full TypeScript coverage
- **State Management**: React useState hooks
- **API Client**: Axios with error handling
- **Animations**: Framer Motion for micro-interactions
- **Charts**: Recharts (AreaChart, RadarChart)
- **UI Components**: Radix UI primitives
- **Styling**: Tailwind CSS with custom design tokens

---

## FastAPI Backend Architecture

### Endpoints

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/api/analyze/{symbol}` | GET | Full analysis + ML prediction | 5-10 sec |
| `/api/quote/{symbol}` | GET | Real-time quote data | <1 sec |
| `/api/historical/{symbol}` | GET | Historical prices (90d) | 1-2 sec |
| `/api/deep-analysis/{symbol}` | POST | CrewAI agent analysis | 30-60 sec |

### Data Flow Fix (2026-02-13)

**Problem:** Frontend wasn't receiving transcript, news, insider data

**Root Cause:** `EarningsOrchestrator` extracted numeric features but discarded raw data

**Solution:**
1. Added new fields to `AnalysisResult` dataclass:
   - `transcript_content: Optional[str]`
   - `news_articles: Optional[list]`
   - `insider_transactions: Optional[list]`

2. Modified `orchestrator.py` to populate raw data before creating result

3. Updated `analysis.py` to map fields for frontend:
   - `site` → `source` (news articles)
   - Added `value` field to insider trades

**Result:** Frontend now displays all data correctly ✅

---

## ML Models - Training Data & Performance

### Training Dataset
- **Source**: Real FMP API data
- **Size**: 1,050 earnings quarters
- **Period**: 2021-2026
- **Symbols**: AAPL, NVDA, MSFT, GOOGL, AMZN, META, TSLA, etc.
- **Features**: 48 (Financial 31, CEO Tone 17, Social calculated)

### Labels (Based on 5-day post-earnings price change)
- **Growth**: Price increased > 5%
- **Risk**: Price decreased > 5%
- **Stagnation**: Price change between -5% and +5%

### Model Performance (Latest Training: 2026-02-13)

| Model | CV Accuracy | Train Accuracy | F1 Score | Status |
|-------|-------------|-----------------|----------|--------|
| **Neural Network (MLP)** | 53.86% | 60.36% | 0.506 | **BEST** ✅ |
| LightGBM | 51.43% | 100.00% | 1.000 | Overfitting |
| Random Forest | 51.00% | 95.95% | 0.960 | Overfitting |
| XGBoost | 50.00% | 100.00% | 1.000 | Severe overfitting |
| Logistic Regression | 39.86% | 49.52% | 0.510 | Worst |

### Version Fix (2026-02-13)
**Problem:** Models trained with sklearn 1.6.1, running with 1.8.0
- Error: `numpy.random._mt19937.MT19937 is not a known BitGenerator`

**Solution:** Retrained all models with current environment
- sklearn 1.6.1
- numpy 1.26.4
- Models now load and predict correctly ✅

---

## Golden Triangle (Unchanged)

| Vector | Weight | Source | Features |
|--------|--------|--------|----------|
| Hard Data | 40% | FMP `/stable/` API | EPS/revenue surprises, margins, analyst consensus, insider sentiment |
| CEO Tone | 35% | FMP Transcripts + NLP | Confidence score, sentiment, guidance, uncertainty (VADER + TextBlob) |
| Social | 25% | **FMP Stock News + NLP** | News sentiment analyzed via VADER + TextBlob |

**Total Features:** 48
**ML Models:** 5 (Neural Net is best)
**Prediction:** Growth / Risk / Stagnation

---

## CrewAI Agents (Optional - Unchanged)

| Agent | Role | Cost |
|-------|------|------|
| Scout Agent | Financial + CEO tone | Included |
| Social Listener | News sentiment | Included |
| Fusion Agent | Synthesizer | Included |
| **Total** | Deep Analysis | ~$0.01-0.02 |

**Model:** GPT-4o-mini (cost-efficient)
**Status:** Optional - user must enable "Deep Analysis"

---

## API Keys Required
- [x] FMP_API_KEY (Ultimate subscription)
- [x] OPENAI_API_KEY (for CrewAI only)

---

## FMP API Endpoints (Unchanged)

### Working Endpoints ✅
- `quote`, `earnings-surprises`, `income-statement`
- `analyst-estimates`, `price-target`, `insider-trading`
- `earnings-call-transcript`, `news/stock-latest`
- `historical-price-eod/full`

### Broken Endpoints ❌
- `social-sentiment` (404), `historical-social-sentiment` (403)
- `institutional-holder` (403), `analyst-stock-recommendations` (403)

**Workaround:** Use `news/stock-latest` + NLP for social sentiment

---

## Deployment

### Local Development
```bash
# Terminal 1 - Backend
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Docker (TODO - needs update)
```bash
docker-compose up --build
```

**Note:** Docker config needs update for React + FastAPI setup

---

## Project Structure (Updated)

```
earnings_hunter/
├── api/                       # FastAPI backend (NEW)
│   ├── main.py
│   └── routers/
│       ├── analysis.py
│       ├── quote.py
│       └── historical.py
│
├── frontend/                  # React frontend (NEW)
│   ├── src/
│   │   ├── components/
│   │   ├── services/api.ts
│   │   ├── types/index.ts
│   │   └── App.tsx
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   └── package.json
│
├── src/                       # Python ML pipeline (PRESERVED)
│   ├── data_ingestion/
│   ├── feature_engineering/
│   ├── ml/
│   └── agents/
│
├── config/settings.py
├── scripts/
│   ├── collect_training_data.py
│   ├── train_model.py
│   └── run_analysis.py
│
├── data/
│   ├── models/               # 5 trained models (retrained 2026-02-13)
│   └── training/             # 1,050 earnings quarters
│
├── app/                       # Legacy Streamlit (DEPRECATED)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── CLAUDE.md                  # This file
└── README.md
```

---

## Decisions Made

1. **React + FastAPI** - Modern stack, better UX than Streamlit
2. **FMP Ultimate** - Unlimited API access
3. **FMP `/stable/` API** - Most reliable endpoints
4. **Stock News + NLP** - Social sentiment from news (VADER 60% + TextBlob 40%)
5. **48 Features** - All extracted by FeaturePipeline, fed to ML
6. **5 ML Models** - Consensus voting, Neural Net is best
7. **Real Training Data** - 1,050 quarters from FMP (2021-2026)
8. **5% Threshold** - Growth/Risk classification
9. **24-hour Cache** - Analysis results cached
10. **CrewAI Optional** - User-enabled deep analysis (~$0.01-0.02)
11. **TypeScript** - Full type safety in frontend
12. **Tailwind CSS** - Utility-first styling
13. **Recharts** - Production-ready charts

---

## Known Issues & TODOs

### High Priority
- [ ] Update Docker config for React + FastAPI
- [ ] Add deployment guide (Vercel frontend + Python API)
- [ ] Add loading states in frontend
- [ ] Error boundary for React app

### Medium Priority
- [ ] Delete old Streamlit files (app/*)
- [ ] Add frontend tests
- [ ] Add API rate limiting
- [ ] Add caching headers

### Low Priority
- [ ] Add dark/light mode toggle
- [ ] Add mobile responsive tweaks
- [ ] Add export to PDF feature
- [ ] Add compare multiple stocks

---

## Quick Start (Updated)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend
npm install
cd ..

# 3. Setup environment
cp .env.example .env
# Edit .env with FMP_API_KEY and OPENAI_API_KEY

# 4. Start backend (Terminal 1)
python -m uvicorn api.main:app --reload --port 8000

# 5. Start frontend (Terminal 2)
cd frontend
npm run dev

# 6. Open browser
# http://localhost:5173
```

---

## Success Metrics

- ✅ React frontend rendering correctly
- ✅ FastAPI backend serving data
- ✅ ML models making predictions
- ✅ All 48 features extracted
- ✅ Transcript content displayed
- ✅ News articles with sentiment
- ✅ Insider trades shown
- ✅ Charts rendering (price, radar)
- ✅ CrewAI agents available (optional)

---

**Last verified working:** 2026-02-13
**Next review:** After deployment to production
