"""Test the actual stock analysis function used in Streamlit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simulate the Streamlit analysis
from datetime import datetime
from config.settings import get_settings
from src.data_ingestion.fmp_client import FMPClient
from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
from src.ml.predictor import EarningsPredictor

settings = get_settings()

def run_real_analysis(ticker: str):
    """Run the same analysis as the Streamlit page."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {ticker}")
    print(f"{'='*60}")

    fmp_client = FMPClient(api_key=settings.fmp_api_key)
    ticker = ticker.upper().strip()

    result = {
        "symbol": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_source": "FMP Ultimate API",
    }

    # Step 1: Get stock quote
    print("\n1. Fetching stock quote...")
    quote = fmp_client.get_stock_quote(ticker)
    if quote:
        result["current_price"] = quote.price
        result["price_change"] = quote.change
        result["price_change_pct"] = quote.change_percentage or 0
        result["company_name"] = quote.name
        print(f"   {quote.name} - ${quote.price:.2f} ({result['price_change_pct']:+.2f}%)")
    else:
        print("   ERROR: Could not get quote")
        return None

    # Step 2: Get earnings data
    print("\n2. Fetching earnings data...")
    earnings = fmp_client.get_earnings_surprises(ticker)
    if earnings and len(earnings) > 0:
        # Find the most recent earnings with actual data
        latest = None
        for e in earnings:
            if e.eps is not None:
                latest = e
                break
        if latest is None:
            latest = earnings[0]

        result["earnings_date"] = latest.date or "N/A"

        # Financial data
        eps_actual = latest.eps or 0
        eps_estimated = latest.eps_estimated or 0
        revenue_actual = (latest.revenue or 0) / 1e9
        revenue_estimated = (latest.revenue_estimated or 0) / 1e9

        eps_surprise = ((eps_actual - eps_estimated) / abs(eps_estimated) * 100) if eps_estimated else 0
        revenue_surprise = ((revenue_actual - revenue_estimated) / abs(revenue_estimated) * 100) if revenue_estimated else 0

        result["financial_data"] = {
            "eps_actual": eps_actual,
            "eps_estimated": eps_estimated,
            "eps_surprise": eps_surprise,
            "revenue_actual": revenue_actual,
            "revenue_estimated": revenue_estimated,
            "revenue_surprise": revenue_surprise,
            "eps_beat": eps_surprise > 0,
            "revenue_beat": revenue_surprise > 0,
        }
        print(f"   EPS: ${eps_actual:.2f} (surprise: {eps_surprise:+.1f}%)")
        print(f"   Revenue: ${revenue_actual:.2f}B (surprise: {revenue_surprise:+.1f}%)")
    else:
        result["financial_data"] = {}
        print("   No earnings data available")

    # Step 3: Get income statement for margins
    print("\n3. Fetching financial statements...")
    statements = fmp_client.get_income_statement(ticker, period="quarter", limit=4)
    if statements and len(statements) > 0:
        latest_stmt = statements[0]
        revenue = latest_stmt.revenue or 0
        gross_profit = latest_stmt.gross_profit or 0
        operating_income = latest_stmt.operating_income or 0
        net_income = latest_stmt.net_income or 0

        result["financial_data"]["gross_margin"] = (gross_profit / revenue * 100) if revenue else 0
        result["financial_data"]["operating_margin"] = (operating_income / revenue * 100) if revenue else 0
        result["financial_data"]["net_margin"] = (net_income / revenue * 100) if revenue else 0
        print(f"   Gross Margin: {result['financial_data']['gross_margin']:.1f}%")
        print(f"   Operating Margin: {result['financial_data']['operating_margin']:.1f}%")
    else:
        print("   No statement data")

    # Step 4: Get analyst data
    print("\n4. Fetching analyst estimates...")
    analyst_estimates = fmp_client.get_analyst_estimates(ticker)
    if analyst_estimates and len(analyst_estimates) > 0:
        result["financial_data"]["analyst_count"] = analyst_estimates[0].number_analyst_estimated_eps or 0
        print(f"   Analyst count: {result['financial_data']['analyst_count']}")

    price_target = fmp_client.get_price_target(ticker)
    if price_target:
        result["financial_data"]["price_target_avg"] = price_target.target_consensus or 0
        if result.get("current_price") and price_target.target_consensus:
            upside = ((price_target.target_consensus - result["current_price"]) / result["current_price"]) * 100
            result["financial_data"]["price_target_upside"] = upside
            print(f"   Price Target: ${price_target.target_consensus:.2f} (upside: {upside:+.1f}%)")

    # Step 5: Get earnings transcript
    print("\n5. Loading earnings transcript...")
    year = datetime.now().year
    quarter = (datetime.now().month - 1) // 3 + 1

    transcript = fmp_client.get_earnings_call_transcript(ticker, year, quarter)

    result["ceo_analysis"] = {
        "has_transcript": False,
        "confidence_score": 0.5,
        "sentiment_score": 0.5,
        "guidance_sentiment": 0.5,
        "uncertainty_ratio": 0.2,
        "tone_summary": "N/A",
    }

    if not transcript or not transcript.content:
        # Try previous quarter
        prev_q = quarter - 1 if quarter > 1 else 4
        prev_y = year if quarter > 1 else year - 1
        transcript = fmp_client.get_earnings_call_transcript(ticker, prev_y, prev_q)

    if transcript and transcript.content:
        print(f"   Transcript found: {len(transcript.content)} characters")
        analyzer = TranscriptAnalyzer()
        tone_features = analyzer.extract_features(transcript.content)

        result["ceo_analysis"] = {
            "has_transcript": True,
            "confidence_score": tone_features.get("ceo_confidence_score", 0.5),
            "sentiment_score": tone_features.get("ceo_overall_sentiment", 0.5),
            "guidance_sentiment": tone_features.get("ceo_forward_guidance_sentiment", 0.5),
            "uncertainty_ratio": tone_features.get("ceo_uncertainty_ratio", 0.2),
            "tone_summary": tone_features.get("ceo_tone_summary", "Neutral"),
        }
        print(f"   CEO Tone: {result['ceo_analysis']['tone_summary']}")
        print(f"   Confidence: {result['ceo_analysis']['confidence_score']:.0%}")
    else:
        print("   No transcript available")

    # Step 6: Calculate Golden Triangle scores
    print("\n6. Calculating Golden Triangle scores...")
    fin_data = result.get("financial_data", {})
    eps_beat_score = 5 if fin_data.get("eps_beat") else 0
    rev_beat_score = 3 if fin_data.get("revenue_beat") else 0
    margin_score = min(2, fin_data.get("gross_margin", 0) / 50)
    financial_score = min(10, eps_beat_score + rev_beat_score + margin_score)

    ceo_data = result.get("ceo_analysis", {})
    confidence = ceo_data.get("confidence_score", 0.5)
    sentiment = ceo_data.get("sentiment_score", 0.5)
    ceo_score = (confidence + sentiment) * 5

    social_score = 5.0  # Default

    weighted_total = (financial_score * 0.40) + (ceo_score * 0.35) + (social_score * 0.25)

    result["scores"] = {
        "financial": round(financial_score, 1),
        "ceo_tone": round(ceo_score, 1),
        "social": round(social_score, 1),
        "weighted_total": round(weighted_total, 1)
    }

    print(f"   Financial Score: {financial_score:.1f}/10 (40%)")
    print(f"   CEO Tone Score: {ceo_score:.1f}/10 (35%)")
    print(f"   Social Score: {social_score:.1f}/10 (25%)")
    print(f"   Weighted Total: {weighted_total:.1f}/10")

    # Step 7: Run ML prediction
    print("\n7. Running ML models...")
    try:
        predictor = EarningsPredictor()
        predictor.load_models()

        features = {
            "fin_eps_surprise": fin_data.get("eps_surprise", 0) / 100,
            "fin_revenue_surprise": fin_data.get("revenue_surprise", 0) / 100,
            "fin_eps_beat": 1 if fin_data.get("eps_beat") else 0,
            "fin_revenue_beat": 1 if fin_data.get("revenue_beat") else 0,
            "fin_gross_margin": fin_data.get("gross_margin", 0) / 100,
            "fin_insider_sentiment": fin_data.get("insider_sentiment", 0),
            "ceo_confidence_score": ceo_data.get("confidence_score", 0.5),
            "ceo_overall_sentiment": ceo_data.get("sentiment_score", 0.5),
            "ceo_uncertainty_ratio": ceo_data.get("uncertainty_ratio", 0.2),
        }

        prediction_result = predictor.predict(features, mode="consensus")
        result["prediction"] = prediction_result.get("consensus_prediction", "Stagnation")
        result["confidence"] = prediction_result.get("best_model_confidence", 0.5)

        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.0%}")

    except Exception as e:
        if weighted_total >= 7:
            result["prediction"] = "Growth"
            result["confidence"] = 0.7
        elif weighted_total <= 4:
            result["prediction"] = "Risk"
            result["confidence"] = 0.6
        else:
            result["prediction"] = "Stagnation"
            result["confidence"] = 0.5
        print(f"   ML Error: {e}")
        print(f"   Fallback Prediction: {result['prediction']}")

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Symbol: {result['symbol']}")
    print(f"Company: {result.get('company_name', 'N/A')}")
    print(f"Price: ${result.get('current_price', 0):.2f}")
    print(f"Golden Triangle: {result['scores']['weighted_total']}/10")
    print(f"Prediction: {result['prediction']} ({result['confidence']:.0%})")

    return result


if __name__ == "__main__":
    # Test with NVDA
    result = run_real_analysis("NVDA")

    print("\n\n")
    # Test with AAPL
    result = run_real_analysis("AAPL")
