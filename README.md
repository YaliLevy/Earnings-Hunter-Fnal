## The Earnings Hunter

An AI‑powered **earnings analysis dashboard** that combines hard financial data, CEO tone, and news sentiment into a single, modern Streamlit one‑pager.  
The app implements a **“Golden Triangle”** framework (40% Financials + 35% CEO Tone + 25% Social/News) plus optional multi‑agent deep analysis via CrewAI.

---

### Key Features

- **Golden Triangle Scoring**
  - Blends earnings surprises, margins, insider activity, CEO tone, and news sentiment into a single 0–10 score.
- **Modern Single‑Page UI**
  - Streamlit one‑pager with glass‑morphism styling, command‑palette search, responsive layout, and rich Plotly charts.
- **ML Earnings Outlook**
  - Multiple trained models (LightGBM, XGBoost, Random Forest, Logistic Regression, Neural Net) with a consensus prediction:
    - **Growth** / **Stagnation** / **Risk** + confidence.
- **CEO Tone & NLP**
  - Analyzes earnings call transcripts (where available) using a custom NLP pipeline (VADER + TextBlob).
- **News‑Driven Social Sentiment**
  - Uses FMP `news/stock-latest` + NLP instead of broken social‑sentiment endpoints; aggregates bullish/bearish ratios.
- **Optional CrewAI Deep Analysis**
  - Three agents (Scout, Social Listener, Fusion) generate a research‑style narrative report on demand.
- **Production‑Ready DevOps**
  - Dockerfile + `docker-compose.yml`, scripts for data collection and model training, and a caching layer.

---

### Tech Stack

- **Frontend / UI**: Streamlit, Plotly
- **Backend / Data**: Python 3.11, FMP `/stable/` API
- **ML**: scikit‑learn, XGBoost, LightGBM, custom trainer & predictor
- **NLP**: VADER, TextBlob
- **Agents**: CrewAI + OpenAI (`gpt-4o-mini`)
- **Infra**: Docker, Docker Compose

---

### Project Structure (High‑Level)

```text
app/
  main.py              # Single-page Streamlit app
  components/          # Charts, metrics, disclaimers, UI building blocks
  config/theme.py      # Colors & CSS (StockFlow-style design system)

config/
  settings.py          # Pydantic settings + env handling

src/
  data_ingestion/      # FMP client + validators
  feature_engineering/ # Financial, CEO tone, social/news features, pipeline
  ml/                  # Model comparison, training, prediction
  agents/              # CrewAI crew, tools, orchestrator, prompts
  utils/               # Logging, caching, helpers

scripts/
  collect_training_data.py
  train_model.py
  run_analysis.py

data/
  models/              # Trained models + metadata
  training/            # Collected training datasets
```

For deeper implementation details, see `CLAUDE.md` (project memory).

---

### Requirements

- **Python**: 3.11 recommended
- **Account / APIs**:
  - FMP **Ultimate** (or compatible) API key
  - OpenAI API key
- **OS**: Developed and tested on Windows 10 + Docker, but should work on macOS/Linux as well.

---

### Environment Variables

Create a `.env` file (you can start from `.env.example`) with at least:

- **Required**
  - `FMP_API_KEY` – your Financial Modeling Prep API key
  - `OPENAI_API_KEY` – your OpenAI API key
- **Optional / Advanced**
  - `LOG_LEVEL` – default `INFO`
  - `CACHE_EXPIRY_HOURS` – default `24`

Reddit credentials are **not** required; social sentiment is derived from stock news.

---

### Quick Start (Local)

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env   # or create .env manually
# Edit .env and set FMP_API_KEY and OPENAI_API_KEY

# 4. Run the Streamlit app
streamlit run app/main.py
```

Then open `http://localhost:8501` in your browser.

---

### Running with Docker

```bash
# Build and start the main app
docker-compose up --build

# App will be available at:
# http://localhost:8501
```

Optional profiles:

- `collect` – run `data-collector` service to gather training data.
- `train` – run `model-trainer` service to retrain models.

Example:

```bash
docker-compose --profile collect up
docker-compose --profile train up
```

---

### How the Golden Triangle Works

- **Hard Financials (40%)**
  - EPS & revenue surprises, analyst targets, margins, insider activity.
- **CEO Tone (35%)**
  - Transcript sentiment, confidence, guidance, and uncertainty scores from NLP.
- **Social / News (25%)**
  - Aggregated sentiment from stock news using VADER + TextBlob.

These three vectors are normalized to 0–10 and combined into a single **Golden Triangle score**, which also feeds the ML prediction and UI visuals (radar chart, anomaly cards, badges, etc.).

---

### CrewAI Deep Analysis

When the **“Deep”** checkbox is enabled in the UI:

- The app runs a CrewAI **three‑agent system**:
  - **Scout Agent** – digs into financials + transcripts.
  - **Social Listener** – analyzes news‑based sentiment.
  - **Fusion Agent** – synthesizes everything (plus ML prediction) into a narrative research summary.
- Typical runtime: **30–60 seconds**, API cost roughly **$0.01–0.02** per deep analysis.

The deep analysis is strictly optional; the fast pipeline runs without any agent cost.

---

### Scripts (CLI Utilities)

- `scripts/collect_training_data.py` – pull historical data from FMP and write CSVs under `data/training/`.
- `scripts/train_model.py` – train multiple ML models and save the best ones under `data/models/`.
- `scripts/run_analysis.py` – run an offline analysis for a given ticker from the command line.

See the script `--help` flags for more options.

---

### Disclaimer

This project is for **educational and research purposes only** and **does not constitute financial advice**.  
Predictions, scores, and visualizations are experimental and should **not** be used as the sole basis for any investment decision.  
Always do your own research and consult a licensed financial professional.

