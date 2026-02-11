# CLAUDE.md - Project Memory & Progress Tracker

> **IMPORTANT**: This file is the project's memory. Update it after EVERY change!
> Last Updated: 2026-02-09 (Single-Page App + Bug Fixes)

## Project Overview
**Name:** The Earnings Hunter
**Goal:** Multi-agent SaaS for earnings analysis using Golden Triangle (40% Financial + 35% CEO Tone + 25% Social)
**Status:** COMPLETE - Ready for testing
**Data Source:** FMP API ONLY (no Reddit/PRAW)

---

## CRITICAL CHANGE: Social Sentiment → Stock News + NLP

### Timeline:
1. **2026-02-05**: Reddit/PRAW → FMP Social Sentiment API
2. **2026-02-09**: FMP Social Sentiment API (broken) → Stock News + NLP Analysis

### Current Architecture (2026-02-09):
| Component | Source | Method |
|-----------|--------|--------|
| Financial Data | FMP `/stable/` API | Direct API calls |
| CEO Tone | FMP Transcripts | NLP analysis (VADER + TextBlob) |
| **Social Sentiment** | **FMP Stock News** | **NLP analysis (VADER + TextBlob)** |

### Why Stock News + NLP?
FMP's social sentiment endpoints return 403/404 errors:
- ❌ `social-sentiment` (404)
- ❌ `historical-social-sentiment` (403)
- ❌ `stock-sentiment-rss-feed` (403)
- ❌ `social-sentiments-trending` (403)
- ❌ `social-sentiments-change` (403)
- ❌ `institutional-holder` (403)
- ❌ `analyst-stock-recommendations` (403)

**Solution:** Use `news/stock-latest` endpoint + VADER/TextBlob NLP

### Benefits:
- ✅ Works with current FMP subscription
- ✅ No additional API keys needed
- ✅ High-quality sentiment from professional news sources
- ✅ VADER + TextBlob combined analysis (60% VADER + 40% TextBlob)

---

## File Creation Progress

### Phase 0: Project Setup
| # | File | Status | Notes |
|---|------|--------|-------|
| 0 | CLAUDE.md | ✅ Done | Project memory file |

### Phase 1: Foundation
| # | File | Status | Notes |
|---|------|--------|-------|
| 1 | requirements.txt | ✅ Done | xgboost, lightgbm, tqdm (NO praw) |
| 2 | .gitignore | ✅ Done | |
| 3 | .env.example | ✅ Done | FMP + OpenAI only (no Reddit) |
| 4 | config/settings.py | ✅ Done | Pydantic BaseSettings + Constants |
| 5 | src/utils/logger.py | ✅ Done | LoggerMixin included |
| 6 | src/utils/cache.py | ✅ Done | 24hr cache with stats |

### Phase 2: Data Ingestion
| # | File | Status | Notes |
|---|------|--------|-------|
| 7 | ~~rate_limiter.py~~ | ❌ DELETED | Was Reddit only |
| 8 | src/data_ingestion/validators.py | ✅ Done | StockNews, EarningsData, etc. |
| 9 | src/data_ingestion/fmp_client.py | ✅ Updated | `/stable/` API, `news/stock-latest` for news |
| 10 | ~~reddit_client.py~~ | ❌ DELETED | Replaced by FMP |

### Phase 3: Feature Engineering
| # | File | Status | Notes |
|---|------|--------|-------|
| 11 | src/feature_engineering/financial_features.py | ✅ Done | 40% weight |
| 12 | src/feature_engineering/sentiment_features.py | ✅ Done | **VADER + TextBlob NLP engine** |
| 13 | src/feature_engineering/transcript_analyzer.py | ✅ Done | 35% weight - CEO tone |
| 14 | src/feature_engineering/social_features.py | ✅ Updated | **Stock News + NLP** (25% weight) |
| 15 | src/feature_engineering/feature_pipeline.py | ✅ Updated | Uses `stock_news` parameter |

### Phase 4: ML Pipeline
| # | File | Status | Notes |
|---|------|--------|-------|
| 16 | src/ml/model_comparison.py | ✅ Done | Multi-model comparison |
| 17 | src/ml/trainer.py | ✅ Done | 5% threshold |
| 18 | src/ml/predictor.py | ✅ Done | Prediction modes |

### Phase 5: Agents
| # | File | Status | Notes |
|---|------|--------|-------|
| 19 | src/agents/config/agents.yaml | ✅ Done | Agent definitions |
| 20 | src/agents/config/tasks.yaml | ✅ Done | Task definitions |
| 21 | src/agents/tools/fmp_tools.py | ✅ Done | FMP tools (social endpoints may fail) |
| 22 | ~~reddit_tools.py~~ | ❌ DELETED | Replaced by FMP tools |
| 23 | src/agents/orchestrator.py | ✅ Updated | Uses `stock_news` + NLP for social |
| 24 | src/agents/insight_generator.py | ✅ Done | Research insight |
| 25 | src/agents/crew.py | ✅ Done | CrewAI + GPT-4o-mini |

### Phase 6: Streamlit Dashboard (ONEPAGER)
| # | File | Status | Notes |
|---|------|--------|-------|
| 26 | app/components/disclaimer.py | ✅ Done | Modal, banner, footer |
| 27 | app/components/charts.py | ✅ Done | Plotly visualizations |
| 28 | app/components/metrics.py | ✅ Done | KPI cards |
| 29 | app/main.py | ✅ **REWRITTEN** | **Single-Page App (Onepager)** - All analysis in one page |
| 30 | ~~app/pages/1_Dashboard.py~~ | ❌ DELETED | Merged into main.py |
| 31 | ~~app/pages/2_Stock_Analysis.py~~ | ❌ DELETED | Merged into main.py |
| 32 | ~~app/pages/3_Historical.py~~ | ❌ DELETED | Merged into main.py |

### Phase 7: Scripts & DevOps
| # | File | Status | Notes |
|---|------|--------|-------|
| 33 | scripts/collect_training_data.py | ✅ Done | Real data |
| 34 | scripts/train_model.py | ✅ Done | Multi-model |
| 35 | scripts/run_analysis.py | ✅ Done | CLI |
| 36 | Dockerfile | ✅ Done | Python 3.11-slim |
| 37 | docker-compose.yml | ✅ Done | Deployment |
| 38 | .dockerignore | ✅ Done | |

---

## Completed Actions Log
<!-- Add new entries at the TOP -->

| Timestamp | Action | Files Changed | Notes |
|-----------|--------|---------------|-------|
| 2026-02-09 | **CONVERTED to ONEPAGER** | app/main.py (rewritten), app/pages/* (deleted) | Single-page app, fixed Golden Triangle, news display, agent report, price chart |
| 2026-02-09 | **FIXED** API Endpoints + NLP Sentiment | fmp_client.py, orchestrator.py, social_features.py, feature_pipeline.py | Stock news + VADER/TextBlob for social sentiment |
| 2026-02-09 | **ADDED** CrewAI Integration | crew.py, 2_Stock_Analysis.py | 3 agents, Deep Analysis checkbox, GPT-4o-mini |
| 2026-02-05 | **MIGRATED** Reddit → FMP Social | Multiple files | Deleted reddit_client, rate_limiter, reddit_tools |
| 2026-02-05 | Created Docker files | Dockerfile, docker-compose.yml | Deployment ready |
| 2026-02-05 | Created scripts | scripts/*.py | Data collection, training |
| 2026-02-05 | Created agents module | src/agents/*.py | CrewAI crew |
| 2026-02-05 | Created Streamlit dashboard | app/**/*.py | All pages |
| 2026-02-05 | Created ML module | src/ml/*.py | Multi-model |
| 2026-02-04 | Created feature engineering | src/feature_engineering/*.py | Golden Triangle |
| 2026-02-04 | Created data ingestion | src/data_ingestion/*.py | FMP client |
| 2026-02-04 | Created foundation files | config/, src/utils/ | Settings, logger |

---

## Single-Page App (Onepager) - 2026-02-09

### Why Onepager?
- Simpler user experience - no navigation needed
- All analysis on one page
- Mobile-friendly design
- Faster loading

### Bugs Fixed in Onepager:
| Bug | Problem | Solution |
|-----|---------|----------|
| Golden Triangle HTML | Displayed raw HTML code instead of progress bars | Use native Streamlit `st.metric()` + `st.progress()` |
| News Articles | Not displaying analyzed articles | Store articles in `result["news_articles"]`, display with sentiment icons |
| Agent Report | Not rendered properly | Added `unsafe_allow_html=True` to `st.markdown()` |
| Price Chart | Empty/not loading | Fetch historical prices from FMP, create Plotly candlestick |
| Multi-Page Navigation | Confusing for users | Converted to single-page app |

### Key Features of Onepager:
- Search bar with ticker input
- Prediction banner (Growth/Risk/Stagnation)
- Golden Triangle scores with native progress bars
- 3-month candlestick price chart
- Tabbed content (Summary, Financial, CEO Tone, News)
- Optional AI Agent deep analysis
- Disclaimer footer

---

## Decisions Made

1. **FMP Ultimate** - No rate limiting (unlimited API access)
2. **FMP `/stable/` API** - Most reliable endpoints, avoid v3/v4
3. **Stock News + NLP** - Social sentiment from news (VADER 60% + TextBlob 40%)
4. **Target Threshold** - 5% for Growth/Risk classification
5. **Training Data** - Real historical data only
6. **Caching** - 24-hour cache for analysis results
7. **Multi-Model** - Train 5 models, use best/consensus
8. **Disclaimer** - Modal + Banner + Footer on EVERY page
9. **CrewAI** - Optional deep analysis using GPT-4o-mini (cost-efficient)
10. **No institutional-holder** - Endpoint returns 403, removed from pipeline

---

## CrewAI Agents (Phase 6 - Added 2026-02-09)

### Architecture: Hybrid Pipeline
```
Fast Pipeline (always runs, FREE)     Deep Analysis (optional, ~$0.01-0.02)
         |                                         |
   FMP Data Fetch                            CrewAI Agents
   TranscriptAnalyzer                        Scout Agent
   ML Models                                 Social Listener
   Golden Triangle                           Fusion Agent
         |                                         |
   Basic Result (<10 sec)              Research Report (30-60 sec)
```

### Three AI Agents

| Agent | Role | Weight | Tools |
|-------|------|--------|-------|
| **Scout Agent** | Financial Data + CEO Tone Analyst | 75% | `fetch_earnings_surprises`, `fetch_income_statement`, `fetch_analyst_data`, `fetch_insider_activity`, `fetch_earnings_transcript` |
| **Social Listener** | Street Sentiment Analyst | 25% | `fetch_stock_news` (social endpoints broken) |
| **Fusion Agent** | Chief Investment Analyst (Synthesizer) | - | No tools (receives other agents' output + ML prediction) |

> ⚠️ **Note:** Social sentiment APIs (`fetch_social_sentiment`, etc.) return 403/404. The pipeline uses `news/stock-latest` + NLP instead.

### Agent Workflow
```
User Request: "Analyze NVDA"
         |
   [Fast Pipeline - FREE]
   FMP Data + ML Prediction
         |
   Basic Result Ready
         |
   [If Deep Analysis checked]
         |
   Scout Agent
   - Fetches financial data
   - Analyzes earnings transcript
   - Generates financial + CEO tone report
         |
   Social Listener Agent
   - Fetches social sentiment
   - Analyzes Twitter/StockTwits
   - Generates sentiment report
         |
   Fusion Agent
   - Receives all reports + ML prediction
   - Applies Golden Triangle weights
   - Generates comprehensive research insight
         |
   AI Agent Report
```

### Cost & Performance
| Analysis Type | Time | Cost | Use Case |
|--------------|------|------|----------|
| Basic (Fast Pipeline) | ~10 sec | FREE | Quick checks, most users |
| Deep (CrewAI Agents) | 30-60 sec | ~$0.01-0.02 | Detailed research reports |

### Files Changed for CrewAI Integration
- `src/agents/crew.py` - Removed Reddit import, uses FMP tools, GPT-4o-mini model
- `app/pages/2_Stock_Analysis.py` - Added "Deep Analysis" checkbox, `run_agent_analysis()` function, "AI Agent Report" tab

---

## API Keys Required
- [x] FMP_API_KEY (Ultimate subscription)
- [x] OPENAI_API_KEY
- [ ] ~~REDDIT_CLIENT_ID~~ (NOT NEEDED - using FMP)
- [ ] ~~REDDIT_CLIENT_SECRET~~ (NOT NEEDED - using FMP)

---

## FMP API Endpoints (Updated 2026-02-09)

### Working Endpoints ✅
| Endpoint | API Version | Used For |
|----------|-------------|----------|
| `quote` | `/stable/` | Current stock price, market cap |
| `earnings-surprises` | `/stable/` | EPS/revenue actual vs estimated |
| `income-statement` | `/stable/` | Financial statements |
| `analyst-estimates` | `/stable/` | Future EPS/revenue estimates |
| `price-target` | `/stable/` | Analyst price targets |
| `insider-trading` | `/stable/` | Insider buy/sell transactions |
| `earnings-call-transcript` | `/stable/` | CEO tone analysis |
| `**news/stock-latest**` | `/stable/` | **Stock news for NLP sentiment** |
| `earnings-calendar` | `/stable/` | Upcoming earnings dates |
| `historical-price-eod/full` | `/stable/` | Historical prices |

### Broken Endpoints ❌ (403/404 errors)
| Endpoint | Error | Workaround |
|----------|-------|------------|
| `social-sentiment` | 404 | Use `news/stock-latest` + NLP |
| `historical-social-sentiment` | 403 | Use `news/stock-latest` + NLP |
| `stock-sentiment-rss-feed` | 403 | Use `news/stock-latest` + NLP |
| `social-sentiments-trending` | 403 | Not needed |
| `social-sentiments-change` | 403 | Not needed |
| `institutional-holder` | 403 | Removed from pipeline |
| `analyst-stock-recommendations` | 403 | Use `analyst-estimates` instead |

---

## Golden Triangle (Updated 2026-02-09)

| Vector | Weight | Source | Features |
|--------|--------|--------|----------|
| Hard Data | 40% | FMP `/stable/` API | EPS/revenue surprises, margins, analyst consensus, insider sentiment |
| CEO Tone | 35% | FMP Transcripts + NLP | Confidence score, sentiment, guidance, uncertainty (VADER + TextBlob) |
| Social | 25% | **FMP Stock News + NLP** | News sentiment analyzed via VADER + TextBlob, hype index, bullish/bearish ratio |

### Social Sentiment Calculation (NLP-based):
```python
# Combined sentiment = 60% VADER compound + 40% TextBlob polarity
combined = (vader_compound * 0.6) + (textblob_polarity * 0.4)  # Range: -1 to +1
bullish_pct = (combined + 1) / 2 * 100  # Convert to 0-100%
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with FMP_API_KEY and OPENAI_API_KEY only

# 3. Run the Streamlit app
streamlit run app/main.py

# 4. (Optional) Docker
docker-compose up --build
```

---

## Project Structure

```
earnings_hunter/
├── CLAUDE.md              # This file - project memory
├── requirements.txt       # NO praw, has crewai
├── .env.example          # FMP + OpenAI only
├── config/
│   └── settings.py
├── src/
│   ├── data_ingestion/
│   │   ├── fmp_client.py     # /stable/ API, news/stock-latest
│   │   └── validators.py     # StockNews, EarningsData, etc.
│   ├── feature_engineering/
│   │   ├── financial_features.py   # 40% weight
│   │   ├── sentiment_features.py   # VADER + TextBlob NLP
│   │   ├── transcript_analyzer.py  # 35% weight - CEO tone
│   │   ├── social_features.py      # 25% weight - News + NLP
│   │   └── feature_pipeline.py     # Golden Triangle orchestration
│   ├── ml/
│   │   ├── model_comparison.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   └── agents/
│       ├── config/
│       │   ├── agents.yaml
│       │   └── tasks.yaml
│       ├── tools/
│       │   └── fmp_tools.py     # CrewAI tools
│       ├── orchestrator.py      # Main analysis pipeline
│       ├── insight_generator.py
│       └── crew.py              # CrewAI 3-agent system
├── app/
│   ├── main.py                # SINGLE-PAGE APP (Onepager) - All functionality here
│   ├── components/
│   │   ├── charts.py
│   │   ├── metrics.py
│   │   └── disclaimer.py
│   └── config/
│       └── theme.py           # Color palette & CSS
├── scripts/
│   ├── collect_training_data.py
│   ├── train_model.py
│   └── run_analysis.py
├── data/
│   └── models/              # Trained ML models
└── Dockerfile
```
