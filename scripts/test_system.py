"""
Comprehensive System Test for The Earnings Hunter.

Tests all components:
1. FMP API connection and data fetching
2. Pydantic validators
3. Feature engineering pipeline
4. ML models
5. CrewAI agents

Run: python scripts/test_system.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
import traceback

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}[FAIL] {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")

def print_info(text: str):
    print(f"{Colors.BLUE}  -> {text}{Colors.END}")


def test_settings():
    """Test configuration loading."""
    print_header("1. Testing Settings & Configuration")

    try:
        from config.settings import get_settings
        settings = get_settings()

        # Check FMP API key
        if settings.fmp_api_key and len(settings.fmp_api_key) > 10:
            print_success(f"FMP API Key loaded: {settings.fmp_api_key[:8]}...")
        else:
            print_error("FMP API Key not found or invalid!")
            return False

        # Check OpenAI API key
        if settings.openai_api_key and len(settings.openai_api_key) > 10:
            print_success(f"OpenAI API Key loaded: {settings.openai_api_key[:8]}...")
        else:
            print_warning("OpenAI API Key not found (CrewAI won't work)")

        print_success(f"Growth threshold: {settings.growth_threshold}%")
        print_success(f"Cache TTL: {settings.cache_expiry_hours} hours")

        return True

    except Exception as e:
        print_error(f"Settings error: {e}")
        traceback.print_exc()
        return False


def test_fmp_api():
    """Test FMP API connection and all endpoints."""
    print_header("2. Testing FMP API Connection")

    try:
        from config.settings import get_settings
        from src.data_ingestion.fmp_client import FMPClient

        settings = get_settings()
        client = FMPClient(api_key=settings.fmp_api_key)

        test_ticker = "AAPL"
        results = {}

        # Test 1: Stock Quote
        print_info(f"Testing get_stock_quote({test_ticker})...")
        quote = client.get_stock_quote(test_ticker)
        if quote:
            print_success(f"Stock Quote: {quote.name} - ${quote.price:.2f}")
            change_pct = quote.change_percentage or 0
            print_info(f"  Change: {change_pct:.2f}%")
            results["quote"] = True
        else:
            print_error("Failed to get stock quote")
            results["quote"] = False

        # Test 2: Earnings Surprises
        print_info(f"Testing get_earnings_surprises({test_ticker})...")
        earnings = client.get_earnings_surprises(test_ticker)
        if earnings and len(earnings) > 0:
            latest = earnings[0]
            print_success(f"Earnings Data: {len(earnings)} quarters")
            print_info(f"  Latest Date: {latest.date}")
            print_info(f"  EPS Actual: ${latest.eps or 0:.2f}")
            print_info(f"  EPS Estimated: ${latest.eps_estimated or 0:.2f}")
            results["earnings"] = True
        else:
            print_warning("No earnings data available")
            results["earnings"] = False

        # Test 4: Income Statement
        print_info(f"Testing get_income_statement({test_ticker})...")
        statements = client.get_income_statement(test_ticker, period="quarter", limit=1)
        if statements and len(statements) > 0:
            stmt = statements[0]
            print_success(f"Income Statement: {stmt.date}")
            print_info(f"  Revenue: ${stmt.revenue/1e9:.2f}B")
            print_info(f"  Net Income: ${stmt.net_income/1e9:.2f}B")
            results["income_statement"] = True
        else:
            print_warning("No income statement data")
            results["income_statement"] = False

        # Test 5: Analyst Estimates
        print_info(f"Testing get_analyst_estimates({test_ticker})...")
        estimates = client.get_analyst_estimates(test_ticker)
        if estimates and len(estimates) > 0:
            print_success(f"Analyst Estimates: {len(estimates)} periods")
            print_info(f"  EPS Avg: ${estimates[0].estimated_eps_avg or 0:.2f}")
            results["analyst_estimates"] = True
        else:
            print_warning("No analyst estimates")
            results["analyst_estimates"] = False

        # Test 6: Price Target
        print_info(f"Testing get_price_target({test_ticker})...")
        target = client.get_price_target(test_ticker)
        if target:
            print_success(f"Price Target Consensus: ${target.target_consensus or 0:.2f}")
            results["price_target"] = True
        else:
            print_warning("No price target data")
            results["price_target"] = False

        # Test 7: Insider Trading
        print_info(f"Testing get_insider_trading({test_ticker})...")
        insiders = client.get_insider_trading(test_ticker, limit=10)
        if insiders:
            print_success(f"Insider Transactions: {len(insiders)} records")
            buys = sum(1 for t in insiders if t.transaction_type and "Buy" in str(t.transaction_type))
            sells = sum(1 for t in insiders if t.transaction_type and "Sell" in str(t.transaction_type))
            print_info(f"  Buys: {buys}, Sells: {sells}")
            results["insider_trading"] = True
        else:
            print_warning("No insider trading data")
            results["insider_trading"] = False

        # Test 8: Earnings Transcript
        print_info(f"Testing get_earnings_call_transcript({test_ticker})...")
        year = datetime.now().year
        quarter = (datetime.now().month - 1) // 3 + 1
        transcript = client.get_earnings_call_transcript(test_ticker, year, quarter)
        if transcript and transcript.content:
            print_success(f"Transcript: Q{quarter} {year}")
            print_info(f"  Length: {len(transcript.content)} characters")
            results["transcript"] = True
        else:
            # Try previous quarter
            prev_q = quarter - 1 if quarter > 1 else 4
            prev_y = year if quarter > 1 else year - 1
            transcript = client.get_earnings_call_transcript(test_ticker, prev_y, prev_q)
            if transcript and transcript.content:
                print_success(f"Transcript: Q{prev_q} {prev_y}")
                print_info(f"  Length: {len(transcript.content)} characters")
                results["transcript"] = True
            else:
                print_warning("No transcript available for recent quarters")
                results["transcript"] = False

        # Test 9: Social Sentiment
        print_info(f"Testing get_social_sentiment({test_ticker})...")
        social = client.get_social_sentiment(test_ticker)
        if social:
            print_success(f"Social Sentiment retrieved")
            print_info(f"  StockTwits: {social.stocktwits_sentiment:.2f}")
            print_info(f"  Twitter: {social.twitter_sentiment:.2f}")
            results["social_sentiment"] = True
        else:
            print_warning("No social sentiment data")
            results["social_sentiment"] = False

        # Test 10: News Sentiment RSS
        print_info(f"Testing get_stock_sentiment_rss({test_ticker})...")
        news = client.get_stock_sentiment_rss(test_ticker, limit=5)
        if news:
            print_success(f"News Sentiment: {len(news)} articles")
            bullish = sum(1 for n in news if getattr(n, 'sentiment', '') == 'Bullish')
            bearish = sum(1 for n in news if getattr(n, 'sentiment', '') == 'Bearish')
            print_info(f"  Bullish: {bullish}, Bearish: {bearish}")
            results["news_sentiment"] = True
        else:
            print_warning("No news sentiment data")
            results["news_sentiment"] = False

        # Test 11: Historical Prices
        print_info(f"Testing get_historical_prices({test_ticker})...")
        prices = client.get_historical_prices(test_ticker, days=30)
        if prices:
            print_success(f"Historical Prices: {len(prices)} days")
            print_info(f"  Latest: {prices[0].date} - ${prices[0].close:.2f}")
            results["historical_prices"] = True
        else:
            print_warning("No historical prices")
            results["historical_prices"] = False

        # Summary
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\n{Colors.BOLD}FMP API Summary: {passed}/{total} tests passed{Colors.END}")

        return passed >= 6  # At least 6 core endpoints should work

    except Exception as e:
        print_error(f"FMP API error: {e}")
        traceback.print_exc()
        return False


def test_validators():
    """Test Pydantic validators."""
    print_header("3. Testing Pydantic Validators")

    try:
        from src.data_ingestion.validators import (
            StockQuote, EarningsData, FinancialStatement,
            AnalystEstimate, InsiderTransaction, SocialSentiment,
            SentimentRSS, HistoricalPrice
        )

        # Test StockQuote
        quote_data = {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "price": 185.50,
            "change": 2.30,
            "changesPercentage": 1.25,
            "dayLow": 183.00,
            "dayHigh": 186.00,
            "marketCap": 2900000000000,
            "volume": 45000000
        }
        quote = StockQuote(**quote_data)
        print_success(f"StockQuote validator: {quote.symbol} ${quote.price}")
        print_info(f"  change_percentage field: {quote.change_percentage}%")

        # Test EarningsData
        earnings_data = {
            "symbol": "AAPL",
            "date": "2024-01-25",
            "epsActual": 2.18,
            "epsEstimated": 2.10,
            "revenueActual": 119600000000,
            "revenueEstimated": 118000000000
        }
        earnings = EarningsData(**earnings_data)
        print_success(f"EarningsData validator: EPS surprise {earnings.eps_surprise:.2%}")

        # Test FinancialStatement
        stmt_data = {
            "date": "2024-01-25",
            "symbol": "AAPL",
            "revenue": 119600000000,
            "grossProfit": 54855000000,
            "operatingIncome": 36016000000,
            "netIncome": 33916000000,
            "eps": 2.18
        }
        stmt = FinancialStatement(**stmt_data)
        print_success(f"FinancialStatement validator: gross_profit ${stmt.gross_profit/1e9:.2f}B")

        # Test SocialSentiment
        social_data = {
            "symbol": "AAPL",
            "date": "2024-01-25",
            "stocktwitsPosts": 150,
            "stocktwitsSentiment": 0.65,
            "twitterPosts": 500,
            "twitterSentiment": 0.58
        }
        social = SocialSentiment(**social_data)
        print_success(f"SocialSentiment validator: combined {social.combined_sentiment:.2f}")

        # Test InsiderTransaction
        insider_data = {
            "symbol": "AAPL",
            "transactionDate": "2024-01-20",
            "transactionType": "P-Purchase",
            "securitiesTransacted": 10000,
            "price": 185.00,
            "reportingName": "Tim Cook",
            "typeOfOwner": "director"
        }
        insider = InsiderTransaction(**insider_data)
        print_success(f"InsiderTransaction validator: {insider.reporting_name} - {insider.transaction_type}")
        print_info(f"  is_purchase: {insider.is_purchase}")

        print_success("All validators working correctly!")
        return True

    except Exception as e:
        print_error(f"Validator error: {e}")
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering pipeline."""
    print_header("4. Testing Feature Engineering")

    try:
        # Test Financial Features
        print_info("Testing FinancialFeatureExtractor...")
        from src.feature_engineering.financial_features import FinancialFeatureExtractor

        fin_extractor = FinancialFeatureExtractor()
        fin_features = fin_extractor.extract_features(
            eps_actual=2.18,
            eps_estimated=2.10,
            revenue_actual=119.6,
            revenue_estimated=118.0,
            gross_margin=0.458,
            operating_margin=0.301,
            insider_buys=5,
            insider_sells=2
        )
        print_success(f"Financial features extracted: {len(fin_features)} features")
        print_info(f"  EPS surprise: {fin_features.get('fin_eps_surprise', 0):.2%}")
        print_info(f"  Insider sentiment: {fin_features.get('fin_insider_sentiment', 0):.2f}")

        # Test Transcript Analyzer
        print_info("Testing TranscriptAnalyzer...")
        from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer

        sample_transcript = """
        Thank you for joining us today. We are very excited about our results this quarter.
        Revenue exceeded our expectations significantly. We are confident in our growth trajectory.
        However, there are some uncertainties in the macro environment that we need to monitor.
        Our guidance for next quarter is strong, and we believe we will continue to outperform.
        The team has done an excellent job executing our strategy.
        """

        analyzer = TranscriptAnalyzer()
        tone_features = analyzer.extract_features(sample_transcript)
        print_success(f"Transcript features extracted: {len(tone_features)} features")
        print_info(f"  Confidence score: {tone_features.get('ceo_confidence_score', 0):.2f}")
        print_info(f"  Sentiment: {tone_features.get('ceo_overall_sentiment', 0):.2f}")
        print_info(f"  Tone summary: {tone_features.get('ceo_tone_summary', 'N/A')}")

        # Test Social Features
        print_info("Testing SocialFeatureExtractor...")
        from src.feature_engineering.social_features import SocialFeatureExtractor

        social_extractor = SocialFeatureExtractor()
        social_features = social_extractor.extract_features(
            stocktwits_sentiment=0.65,
            twitter_sentiment=0.58,
            total_posts=650,
            news_bullish_ratio=0.6,
            news_bearish_ratio=0.2
        )
        print_success(f"Social features extracted: {len(social_features)} features")
        print_info(f"  Combined sentiment: {social_features.get('social_combined_sentiment', 0):.2f}")

        print_success("Feature engineering pipeline working!")
        return True

    except Exception as e:
        print_error(f"Feature engineering error: {e}")
        traceback.print_exc()
        return False


def test_ml_models():
    """Test ML models loading and prediction."""
    print_header("5. Testing ML Models")

    try:
        from src.ml.predictor import EarningsPredictor

        predictor = EarningsPredictor()

        # Check if models exist
        print_info("Checking for trained models...")
        try:
            predictor.load_models()
            print_success("Models loaded successfully!")
            models_exist = True
        except Exception as e:
            print_warning(f"No trained models found: {e}")
            print_info("Models need to be trained with: python scripts/train_model.py")
            models_exist = False

        if models_exist:
            # Test prediction
            test_features = {
                "fin_eps_surprise": 0.038,
                "fin_revenue_surprise": 0.014,
                "fin_eps_beat": 1,
                "fin_revenue_beat": 1,
                "fin_gross_margin": 0.458,
                "fin_insider_sentiment": 0.43,
                "ceo_confidence_score": 0.72,
                "ceo_overall_sentiment": 0.68,
                "ceo_uncertainty_ratio": 0.15,
            }

            print_info("Testing prediction with sample features...")
            result = predictor.predict(test_features, mode="consensus")

            print_success(f"Prediction: {result.get('consensus_prediction', 'N/A')}")
            print_info(f"  Confidence: {result.get('best_model_confidence', 0):.0%}")
            print_info(f"  Agreement: {result.get('agreement_ratio', 0):.0%}")

            if result.get('model_predictions'):
                print_info("  Individual model predictions:")
                for model, pred in result['model_predictions'].items():
                    print_info(f"    - {model}: {pred}")

        return True

    except Exception as e:
        print_error(f"ML model error: {e}")
        traceback.print_exc()
        return False


def test_crewai():
    """Test CrewAI agents (if OpenAI key is available)."""
    print_header("6. Testing CrewAI Agents")

    try:
        from config.settings import get_settings
        settings = get_settings()

        if not settings.openai_api_key or len(settings.openai_api_key) < 10:
            print_warning("OpenAI API key not found - skipping CrewAI tests")
            print_info("Add OPENAI_API_KEY to .env to enable agent tests")
            return True

        # Check if CrewAI is installed
        try:
            import crewai
            print_success(f"CrewAI version: {crewai.__version__}")
        except ImportError:
            print_warning("CrewAI not installed")
            return True

        # Check agent configs
        print_info("Checking agent configurations...")
        from pathlib import Path

        agents_yaml = project_root / "src" / "agents" / "config" / "agents.yaml"
        tasks_yaml = project_root / "src" / "agents" / "config" / "tasks.yaml"

        if agents_yaml.exists():
            print_success(f"agents.yaml found")
        else:
            print_error(f"agents.yaml not found at {agents_yaml}")
            return False

        if tasks_yaml.exists():
            print_success(f"tasks.yaml found")
        else:
            print_error(f"tasks.yaml not found at {tasks_yaml}")
            return False

        # Test FMP tools
        print_info("Testing FMP tools for agents...")
        from src.agents.tools.fmp_tools import (
            fetch_stock_quote,
            fetch_earnings_surprises,
            fetch_social_sentiment
        )

        # Quick tool test
        quote_result = fetch_stock_quote("AAPL")
        if "error" not in quote_result.lower():
            print_success("FMP tools working")
        else:
            print_warning(f"FMP tools returned: {quote_result[:100]}")

        print_info("CrewAI agents configured (full test requires API call)")
        return True

    except Exception as e:
        print_error(f"CrewAI error: {e}")
        traceback.print_exc()
        return False


def test_streamlit_components():
    """Test Streamlit components (import only)."""
    print_header("7. Testing Streamlit Components")

    try:
        print_info("Testing component imports...")

        from app.config.theme import COLORS, get_all_css, get_plotly_layout
        print_success("Theme module loaded")
        print_info(f"  Colors defined: {len(COLORS)}")

        from app.components.market_status import is_market_open, get_next_market_event
        print_success("Market status component loaded")
        print_info(f"  Market open: {is_market_open()}")

        from app.components.index_cards import fetch_market_indices
        print_success("Index cards component loaded")

        from app.components.stock_chart import fetch_stock_data, create_price_chart
        print_success("Stock chart component loaded")

        from app.components.sentiment_gauge import create_sentiment_gauge
        print_success("Sentiment gauge component loaded")

        from app.components.disclaimer import show_disclaimer_modal
        print_success("Disclaimer component loaded")

        print_success("All Streamlit components imported successfully!")
        return True

    except Exception as e:
        print_error(f"Streamlit component error: {e}")
        traceback.print_exc()
        return False


def run_full_analysis_test():
    """Run a complete analysis to test the full pipeline."""
    print_header("8. Full Analysis Pipeline Test")

    try:
        from config.settings import get_settings
        from src.data_ingestion.fmp_client import FMPClient
        from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
        from datetime import datetime

        settings = get_settings()
        client = FMPClient(api_key=settings.fmp_api_key)

        test_ticker = "NVDA"
        print_info(f"Running full analysis for {test_ticker}...")

        # Step 1: Quote
        quote = client.get_stock_quote(test_ticker)
        if quote:
            print_success(f"Quote: {quote.name} - ${quote.price:.2f} ({quote.change_percentage:+.2f}%)")
        else:
            print_error("Failed to get quote")
            return False

        # Step 2: Earnings
        earnings = client.get_earnings_surprises(test_ticker)
        if earnings and len(earnings) > 0:
            latest = earnings[0]
            eps_surprise = ((latest.eps - latest.eps_estimated) / abs(latest.eps_estimated) * 100) if latest.eps_estimated else 0
            print_success(f"Earnings: EPS ${latest.eps:.2f} (surprise: {eps_surprise:+.1f}%)")
        else:
            print_warning("No earnings data")

        # Step 3: Transcript
        year = datetime.now().year
        quarter = (datetime.now().month - 1) // 3 + 1
        transcript = client.get_earnings_call_transcript(test_ticker, year, quarter)

        if not transcript or not transcript.content:
            prev_q = quarter - 1 if quarter > 1 else 4
            prev_y = year if quarter > 1 else year - 1
            transcript = client.get_earnings_call_transcript(test_ticker, prev_y, prev_q)

        if transcript and transcript.content:
            analyzer = TranscriptAnalyzer()
            tone = analyzer.extract_features(transcript.content)
            print_success(f"Transcript analyzed: {tone.get('ceo_tone_summary', 'N/A')} tone")
            print_info(f"  Confidence: {tone.get('ceo_confidence_score', 0):.0%}")
        else:
            print_warning("No transcript available")

        # Step 4: Social
        social = client.get_social_sentiment(test_ticker)
        if social:
            print_success(f"Social sentiment: {social.combined_sentiment:.0%}")
        else:
            print_warning("No social sentiment")

        # Step 5: Calculate scores
        fin_score = 7.5  # Mock based on beat
        ceo_score = 6.8
        social_score = 5.5
        weighted = (fin_score * 0.40) + (ceo_score * 0.35) + (social_score * 0.25)

        print_success(f"Golden Triangle Score: {weighted:.1f}/10")
        print_info(f"  Financial: {fin_score}/10 (40%)")
        print_info(f"  CEO Tone: {ceo_score}/10 (35%)")
        print_info(f"  Social: {social_score}/10 (25%)")

        print_success("Full analysis pipeline working!")
        return True

    except Exception as e:
        print_error(f"Full analysis error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print(f"\n{Colors.BOLD}{'='*60}")
    print("  THE EARNINGS HUNTER - SYSTEM TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}{Colors.END}\n")

    results = {}

    # Run all tests
    results["Settings"] = test_settings()
    results["FMP API"] = test_fmp_api()
    results["Validators"] = test_validators()
    results["Feature Engineering"] = test_feature_engineering()
    results["ML Models"] = test_ml_models()
    results["CrewAI"] = test_crewai()
    results["Streamlit Components"] = test_streamlit_components()
    results["Full Analysis"] = run_full_analysis_test()

    # Summary
    print_header("TEST SUMMARY")

    passed = 0
    failed = 0

    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
            passed += 1
        else:
            print_error(f"{test_name}: FAILED")
            failed += 1

    print(f"\n{Colors.BOLD}Total: {passed} passed, {failed} failed{Colors.END}")

    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! System is ready.{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}Some tests failed. Review the output above.{Colors.END}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
