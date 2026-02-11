#!/usr/bin/env python3
"""
Training Data Collection Script for The Earnings Hunter.

Collects REAL historical earnings data from FMP Ultimate API including:
- Earnings data (EPS, revenue vs estimates)
- Earnings call transcripts (for CEO tone analysis)
- Historical prices (for calculating actual outcomes)
- Analyst data, insider activity

Uses REAL transcripts - no synthetic data!
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tqdm import tqdm

from src.data_ingestion.fmp_client import FMPClient
from src.feature_engineering.financial_features import FinancialFeatureExtractor
from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.utils.logger import get_logger
from config.settings import get_settings, Constants

logger = get_logger(__name__)
settings = get_settings()

# Default symbols to collect (50+ for good training data)
DEFAULT_SYMBOLS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Semiconductors
    "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "AMAT",
    # Software/Cloud
    "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR", "DDOG",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY",
    # Consumer
    "WMT", "HD", "COST", "NKE", "SBUX", "MCD", "DIS",
    # Industrial
    "CAT", "BA", "HON", "UPS", "MMM", "GE", "LMT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG"
]


def calculate_outcome(
    pre_price: float,
    post_price: float,
    growth_threshold: float = 5.0,
    risk_threshold: float = -5.0
) -> str:
    """
    Calculate outcome label based on price movement.

    Args:
        pre_price: Price before earnings
        post_price: Price 3 days after earnings
        growth_threshold: % threshold for Growth label (default 5%)
        risk_threshold: % threshold for Risk label (default -5%)

    Returns:
        Label: "Growth", "Stagnation", or "Risk"
    """
    if pre_price <= 0:
        return "Unknown"

    pct_change = ((post_price - pre_price) / pre_price) * 100

    if pct_change >= growth_threshold:
        return "Growth"
    elif pct_change <= risk_threshold:
        return "Risk"
    else:
        return "Stagnation"


def get_quarter_dates(year: int, quarter: int) -> tuple:
    """Get approximate start and end dates for a quarter."""
    quarter_starts = {
        1: f"{year}-01-01",
        2: f"{year}-04-01",
        3: f"{year}-07-01",
        4: f"{year}-10-01"
    }
    quarter_ends = {
        1: f"{year}-03-31",
        2: f"{year}-06-30",
        3: f"{year}-09-30",
        4: f"{year}-12-31"
    }
    return quarter_starts[quarter], quarter_ends[quarter]


def collect_earnings_data(
    symbols: List[str],
    start_year: int = 2021,
    end_year: int = 2025,
    output_dir: str = "data/training"
) -> pd.DataFrame:
    """
    Collect comprehensive training data for all symbols and quarters.

    Args:
        symbols: List of stock symbols
        start_year: Start year for data collection
        end_year: End year for data collection
        output_dir: Directory to save output

    Returns:
        DataFrame with all training features and labels
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize clients and extractors
    fmp_client = FMPClient(api_key=settings.fmp_api_key)
    financial_extractor = FinancialFeatureExtractor()
    transcript_analyzer = TranscriptAnalyzer()
    feature_pipeline = FeaturePipeline()

    all_data = []
    errors = []

    # Calculate total tasks for progress bar
    quarters = list(range(1, 5))
    years = list(range(start_year, end_year + 1))
    total_tasks = len(symbols) * len(years) * len(quarters)

    logger.info(f"Collecting data for {len(symbols)} symbols, {len(years)} years, {len(quarters)} quarters")
    logger.info(f"Total earnings events to process: {total_tasks}")

    with tqdm(total=total_tasks, desc="Collecting Training Data") as pbar:
        for symbol in symbols:
            for year in years:
                for quarter in quarters:
                    pbar.set_description(f"Processing {symbol} Q{quarter} {year}")

                    try:
                        # Skip future quarters
                        if year == datetime.now().year and quarter > (datetime.now().month - 1) // 3 + 1:
                            pbar.update(1)
                            continue

                        record = collect_single_earnings(
                            fmp_client,
                            financial_extractor,
                            transcript_analyzer,
                            feature_pipeline,
                            symbol,
                            year,
                            quarter
                        )

                        if record:
                            all_data.append(record)
                            logger.debug(f"Collected {symbol} Q{quarter} {year}")

                    except Exception as e:
                        errors.append({
                            "symbol": symbol,
                            "year": year,
                            "quarter": quarter,
                            "error": str(e)
                        })
                        logger.warning(f"Error collecting {symbol} Q{quarter} {year}: {e}")

                    pbar.update(1)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"training_data_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} records to {csv_path}")

    # Save errors log
    if errors:
        errors_path = output_path / f"collection_errors_{timestamp}.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        logger.info(f"Saved {len(errors)} errors to {errors_path}")

    # Print summary
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Total records collected: {len(df)}")
    print(f"Total errors: {len(errors)}")
    print(f"Symbols processed: {len(symbols)}")
    print(f"Date range: Q1 {start_year} - Q4 {end_year}")

    if len(df) > 0:
        print(f"\nLabel distribution:")
        print(df["label"].value_counts())

        print(f"\nTranscripts collected: {df['has_transcript'].sum()}")
        print(f"Missing transcripts: {len(df) - df['has_transcript'].sum()}")

    print(f"\nOutput saved to: {csv_path}")
    print("="*60)

    return df


def collect_single_earnings(
    fmp_client: FMPClient,
    financial_extractor: FinancialFeatureExtractor,
    transcript_analyzer: TranscriptAnalyzer,
    feature_pipeline: FeaturePipeline,
    symbol: str,
    year: int,
    quarter: int
) -> Optional[Dict[str, Any]]:
    """
    Collect all features for a single earnings event.

    Args:
        fmp_client: FMP API client
        financial_extractor: Financial feature extractor
        transcript_analyzer: Transcript analyzer
        feature_pipeline: Feature pipeline
        symbol: Stock symbol
        year: Year
        quarter: Quarter (1-4)

    Returns:
        Dictionary of features and label, or None if data unavailable
    """
    record = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "collection_date": datetime.now().isoformat()
    }

    # Get earnings data
    earnings = fmp_client.get_earnings_surprises(symbol)
    if not earnings:
        return None

    # Find the earnings for this quarter
    target_earnings = None
    for e in earnings:
        if hasattr(e, "date") and e.date:
            e_date = datetime.strptime(e.date, "%Y-%m-%d")
            e_quarter = (e_date.month - 1) // 3 + 1
            if e_date.year == year and e_quarter == quarter:
                target_earnings = e
                break

    if not target_earnings:
        return None

    record["earnings_date"] = target_earnings.date

    # Extract financial features
    try:
        income_statements = fmp_client.get_income_statement(symbol, period="quarter", limit=8)
        financial_features = financial_extractor.extract_all(
            earnings=target_earnings,
            statements=income_statements
        )
        record.update(financial_features)
    except Exception as e:
        logger.debug(f"Could not extract financial features for {symbol}: {e}")

    # Get analyst data
    try:
        analyst_estimates = fmp_client.get_analyst_estimates(symbol)
        if analyst_estimates:
            record["analyst_count"] = analyst_estimates[0].numberAnalystEstimatedEps if analyst_estimates else 0
    except Exception:
        pass

    # Get insider data
    try:
        insider_data = fmp_client.get_insider_trading(symbol, limit=50)
        if insider_data:
            buys = sum(1 for t in insider_data if t.transactionType and "Buy" in t.transactionType)
            sells = sum(1 for t in insider_data if t.transactionType and "Sell" in t.transactionType)
            total = buys + sells
            record["insider_sentiment"] = (buys - sells) / total if total > 0 else 0
    except Exception:
        pass

    # Get transcript and analyze CEO tone
    try:
        transcript = fmp_client.get_earnings_call_transcript(symbol, year, quarter)
        if transcript and transcript.content:
            record["has_transcript"] = True

            # Analyze CEO tone
            tone_features = transcript_analyzer.extract_features(transcript.content)
            record.update(tone_features)
        else:
            record["has_transcript"] = False
    except Exception as e:
        record["has_transcript"] = False
        logger.debug(f"Could not get transcript for {symbol} Q{quarter} {year}: {e}")

    # Get prices for outcome calculation
    try:
        earnings_date = datetime.strptime(target_earnings.date, "%Y-%m-%d")

        # Price 1 day before earnings
        pre_date = (earnings_date - timedelta(days=1)).strftime("%Y-%m-%d")
        # Price 3 days after earnings
        post_date = (earnings_date + timedelta(days=5)).strftime("%Y-%m-%d")

        prices = fmp_client.get_historical_prices(
            symbol,
            from_date=(earnings_date - timedelta(days=5)).strftime("%Y-%m-%d"),
            to_date=post_date
        )

        if prices:
            # Find pre and post prices
            pre_price = None
            post_price = None

            for p in sorted(prices, key=lambda x: x.date):
                p_date = datetime.strptime(p.date, "%Y-%m-%d")
                if p_date < earnings_date:
                    pre_price = p.close
                elif p_date >= earnings_date + timedelta(days=3):
                    post_price = p.close
                    break

            if pre_price and post_price:
                record["pre_price"] = pre_price
                record["post_price"] = post_price
                record["price_change_pct"] = ((post_price - pre_price) / pre_price) * 100
                record["label"] = calculate_outcome(pre_price, post_price)
            else:
                record["label"] = "Unknown"
        else:
            record["label"] = "Unknown"

    except Exception as e:
        record["label"] = "Unknown"
        logger.debug(f"Could not calculate outcome for {symbol}: {e}")

    return record if record.get("label") != "Unknown" else None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect training data for The Earnings Hunter"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Stock symbols to collect (default: 50 major stocks)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year (default: 2021)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year (default: 2025)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory (default: data/training)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("THE EARNINGS HUNTER - TRAINING DATA COLLECTION")
    print("="*60)
    print(f"Symbols: {len(args.symbols)}")
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Output: {args.output_dir}")
    print("="*60 + "\n")

    # Run collection
    df = collect_earnings_data(
        symbols=args.symbols,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir
    )

    if len(df) < Constants.MIN_TRAINING_SAMPLES:
        print(f"\nWARNING: Only collected {len(df)} samples.")
        print(f"Minimum recommended: {Constants.MIN_TRAINING_SAMPLES}")
        print("Consider expanding symbol list or date range.")


if __name__ == "__main__":
    main()
