#!/usr/bin/env python3
"""
Single Stock Analysis Script for The Earnings Hunter.

Run a full Golden Triangle analysis for a single stock.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import EarningsOrchestrator
from src.utils.logger import get_logger
from config.settings import Constants

logger = get_logger(__name__)


def run_analysis(symbol: str, force_refresh: bool = False) -> dict:
    """
    Run full analysis for a symbol.

    Args:
        symbol: Stock ticker symbol
        force_refresh: Bypass cache if True

    Returns:
        Analysis result dictionary
    """
    print("\n" + "="*60)
    print(f"THE EARNINGS HUNTER - ANALYZING {symbol.upper()}")
    print("="*60)

    orchestrator = EarningsOrchestrator()

    print(f"\nAnalyzing {symbol.upper()}...")
    print("This may take a few moments...\n")

    result = orchestrator.analyze_latest_earnings(symbol, force_refresh=force_refresh)

    if result.status == "error":
        print(f"Error: {result.error}")
        return result.__dict__

    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    print(f"\nSymbol: {result.symbol}")
    print(f"Earnings Date: {result.earnings_date}")
    print(f"Quarter: Q{result.quarter} {result.year}")

    print("\n" + "-"*40)
    print("GOLDEN TRIANGLE SCORES")
    print("-"*40)
    scores = result.golden_triangle_scores
    print(f"Financial (40%):   {scores.get('financial', 'N/A'):.1f}/10")
    print(f"CEO Tone (35%):    {scores.get('ceo_tone', 'N/A'):.1f}/10")
    print(f"Social (25%):      {scores.get('social', 'N/A'):.1f}/10")
    print(f"Weighted Total:    {scores.get('weighted_total', 'N/A'):.1f}/10")

    print("\n" + "-"*40)
    print("PREDICTION")
    print("-"*40)
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.0%}")

    if result.model_agreement:
        agree = result.model_agreement.get("models_agree", 0)
        total = result.model_agreement.get("models_total", 0)
        print(f"Model Agreement: {agree}/{total}")

    print("\n" + "-"*40)
    print("RESEARCH INSIGHT")
    print("-"*40)

    insight = result.insight
    if insight:
        print(f"\nExecutive Summary:")
        print(f"  {insight.executive_summary}")

        print(f"\nKey Highlights:")
        for h in insight.key_highlights:
            print(f"  - {h}")

        print(f"\nShort-Term Outlook:")
        print(f"  {insight.short_term_outlook}")

        print(f"\nLong-Term Outlook:")
        print(f"  {insight.long_term_outlook}")

        print(f"\nKey Risks:")
        for r in insight.key_risks:
            print(f"  - {r}")

        print(f"\nBottom Line:")
        print(f"  {insight.bottom_line}")

    print("\n" + "="*60)
    print("DISCLAIMER")
    print("="*60)
    print(Constants.DISCLAIMER_TEXT)
    print("="*60 + "\n")

    return result.__dict__


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run earnings analysis for a single stock"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., NVDA)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cache and fetch fresh data"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    result = run_analysis(args.symbol, args.force_refresh)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
