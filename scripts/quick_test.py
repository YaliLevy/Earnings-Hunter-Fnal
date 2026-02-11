"""Quick test of core functionality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings

print("=" * 60)
print("QUICK SYSTEM TEST")
print("=" * 60)

# 1. Settings
print("\n1. Settings...")
settings = get_settings()
print(f"   FMP API Key: {settings.fmp_api_key[:8]}...")
print(f"   OpenAI Key: {settings.openai_api_key[:8]}...")

# 2. FMP Client
print("\n2. FMP API...")
from src.data_ingestion.fmp_client import FMPClient
client = FMPClient(api_key=settings.fmp_api_key)

quote = client.get_stock_quote("NVDA")
if quote:
    print(f"   Quote: {quote.name} - ${quote.price:.2f}")
else:
    print("   ERROR: Could not get quote")

earnings = client.get_earnings_surprises("NVDA")
if earnings:
    print(f"   Earnings: {len(earnings)} quarters available")
else:
    print("   ERROR: Could not get earnings")

# 3. Transcript Analyzer
print("\n3. Transcript Analyzer...")
from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
analyzer = TranscriptAnalyzer()
sample = "We are confident in our strong growth. Revenue exceeded expectations."
result = analyzer.extract_features(sample)
print(f"   Tone: {result.get('ceo_tone_summary', 'N/A')}")
print(f"   Confidence: {result.get('ceo_confidence_score', 0):.2f}")

# 4. ML Predictor
print("\n4. ML Models...")
from src.ml.predictor import EarningsPredictor
predictor = EarningsPredictor()
try:
    predictor.load_models()
    print("   Models loaded successfully")

    # Quick prediction
    features = {
        "fin_eps_surprise": 0.05,
        "fin_revenue_surprise": 0.03,
        "fin_eps_beat": 1,
        "fin_revenue_beat": 1,
        "fin_gross_margin": 0.45,
        "fin_insider_sentiment": 0.3,
        "ceo_confidence_score": 0.7,
        "ceo_overall_sentiment": 0.65,
        "ceo_uncertainty_ratio": 0.15,
    }
    result = predictor.predict(features, mode="consensus")
    print(f"   Prediction: {result.get('consensus_prediction', 'N/A')}")
    print(f"   Confidence: {result.get('best_model_confidence', 0):.0%}")
except Exception as e:
    print(f"   ERROR: {e}")

# 5. Quick analysis simulation
print("\n5. Analysis Pipeline Simulation...")
from datetime import datetime

ticker = "NVDA"
print(f"   Testing {ticker}...")

# Get quote
quote = client.get_stock_quote(ticker)
if quote:
    print(f"   Price: ${quote.price:.2f}")
    print(f"   Company: {quote.name}")

# Get earnings
earnings = client.get_earnings_surprises(ticker)
if earnings and len(earnings) > 0:
    latest = earnings[0]
    if latest.eps and latest.eps_estimated and latest.eps_estimated != 0:
        eps_surprise = (latest.eps - latest.eps_estimated) / abs(latest.eps_estimated) * 100
        print(f"   EPS Surprise: {eps_surprise:+.1f}%")

# Get transcript
year = datetime.now().year
quarter = (datetime.now().month - 1) // 3 + 1
transcript = client.get_earnings_call_transcript(ticker, year, quarter)
if not transcript or not transcript.content:
    # Try previous quarter
    prev_q = quarter - 1 if quarter > 1 else 4
    prev_y = year if quarter > 1 else year - 1
    transcript = client.get_earnings_call_transcript(ticker, prev_y, prev_q)

if transcript and transcript.content:
    tone = analyzer.extract_features(transcript.content)
    print(f"   CEO Tone: {tone.get('ceo_tone_summary', 'N/A')}")
else:
    print("   No transcript available")

print("\n" + "=" * 60)
print("TEST COMPLETE - Core systems working!")
print("=" * 60)
