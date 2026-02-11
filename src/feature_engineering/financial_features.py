"""
Financial feature extraction from FMP data.

Hard Data features represent 40% of the Golden Triangle.
"""

from typing import Dict, List, Optional, Any

from src.utils.logger import get_logger
from src.data_ingestion.validators import (
    EarningsData,
    FinancialStatement,
    AnalystEstimate,
    InsiderTransaction,
    InstitutionalHolder,
    PriceTarget,
    StockQuote,
)

logger = get_logger(__name__)


class FinancialFeatureExtractor:
    """
    Extract financial features from FMP data.

    These features represent 40% of the Golden Triangle weighting.
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_prefix = "fin_"

    def extract_earnings_features(
        self,
        earnings: Optional[EarningsData]
    ) -> Dict[str, float]:
        """
        Extract features from earnings data.

        Args:
            earnings: EarningsData object

        Returns:
            Dict of earnings features
        """
        features = {}

        if not earnings:
            return self._empty_earnings_features()

        # EPS features
        if earnings.eps is not None and earnings.eps_estimated is not None:
            eps_estimated = earnings.eps_estimated
            if eps_estimated != 0:
                features["eps_surprise"] = (earnings.eps - eps_estimated) / abs(eps_estimated)
            else:
                features["eps_surprise"] = 0.0
            features["eps_beat"] = 1.0 if earnings.eps > eps_estimated else 0.0
            features["eps_actual"] = earnings.eps
        else:
            features["eps_surprise"] = 0.0
            features["eps_beat"] = 0.0
            features["eps_actual"] = 0.0

        # Revenue features
        if earnings.revenue is not None and earnings.revenue_estimated is not None:
            rev_estimated = earnings.revenue_estimated
            if rev_estimated != 0:
                features["revenue_surprise"] = (earnings.revenue - rev_estimated) / rev_estimated
            else:
                features["revenue_surprise"] = 0.0
            features["revenue_beat"] = 1.0 if earnings.revenue > rev_estimated else 0.0
        else:
            features["revenue_surprise"] = 0.0
            features["revenue_beat"] = 0.0

        # Combined beat
        features["double_beat"] = 1.0 if features["eps_beat"] and features["revenue_beat"] else 0.0

        return features

    def _empty_earnings_features(self) -> Dict[str, float]:
        """Return empty earnings features."""
        return {
            "eps_surprise": 0.0,
            "eps_beat": 0.0,
            "eps_actual": 0.0,
            "revenue_surprise": 0.0,
            "revenue_beat": 0.0,
            "double_beat": 0.0,
        }

    def extract_statement_features(
        self,
        statements: List[FinancialStatement]
    ) -> Dict[str, float]:
        """
        Extract features from financial statements.

        Calculates growth rates and margins.

        Args:
            statements: List of FinancialStatement objects (most recent first)

        Returns:
            Dict of statement features
        """
        features = {}

        if not statements:
            return self._empty_statement_features()

        current = statements[0]

        # Current margins
        features["gross_margin"] = current.gross_margin or 0.0
        features["operating_margin"] = current.operating_margin or 0.0
        features["net_margin"] = current.net_margin or 0.0

        # Quarter-over-Quarter growth (compare to previous quarter)
        if len(statements) >= 2:
            previous = statements[1]
            features["qoq_revenue_growth"] = self._calc_growth(current.revenue, previous.revenue)
            features["qoq_earnings_growth"] = self._calc_growth(current.net_income, previous.net_income)
        else:
            features["qoq_revenue_growth"] = 0.0
            features["qoq_earnings_growth"] = 0.0

        # Year-over-Year growth (compare to 4 quarters ago)
        if len(statements) >= 5:
            yoy = statements[4]
            features["yoy_revenue_growth"] = self._calc_growth(current.revenue, yoy.revenue)
            features["yoy_earnings_growth"] = self._calc_growth(current.net_income, yoy.net_income)
        else:
            features["yoy_revenue_growth"] = 0.0
            features["yoy_earnings_growth"] = 0.0

        # Margin trends (compare to previous quarter)
        if len(statements) >= 2:
            previous = statements[1]
            features["gross_margin_change"] = (current.gross_margin or 0) - (previous.gross_margin or 0)
            features["operating_margin_change"] = (current.operating_margin or 0) - (previous.operating_margin or 0)
        else:
            features["gross_margin_change"] = 0.0
            features["operating_margin_change"] = 0.0

        return features

    def _empty_statement_features(self) -> Dict[str, float]:
        """Return empty statement features."""
        return {
            "gross_margin": 0.0,
            "operating_margin": 0.0,
            "net_margin": 0.0,
            "qoq_revenue_growth": 0.0,
            "qoq_earnings_growth": 0.0,
            "yoy_revenue_growth": 0.0,
            "yoy_earnings_growth": 0.0,
            "gross_margin_change": 0.0,
            "operating_margin_change": 0.0,
        }

    def _calc_growth(self, current: float, previous: float) -> float:
        """Calculate growth rate between two values."""
        if previous == 0:
            return 0.0
        return (current - previous) / abs(previous)

    def extract_analyst_features(
        self,
        estimates: List[AnalystEstimate],
        price_target: Optional[PriceTarget],
        current_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract features from analyst data.

        Args:
            estimates: List of AnalystEstimate objects
            price_target: PriceTarget object
            current_price: Current stock price

        Returns:
            Dict of analyst features
        """
        features = {}

        # Analyst estimates
        if estimates:
            latest = estimates[0]
            features["analyst_count_eps"] = float(latest.number_analyst_estimated_eps or 0)
            features["analyst_count_revenue"] = float(latest.number_analyst_estimated_revenue or 0)

            # Estimate range (uncertainty indicator)
            if latest.estimated_eps_high and latest.estimated_eps_low:
                features["eps_estimate_range"] = latest.estimated_eps_high - latest.estimated_eps_low
            else:
                features["eps_estimate_range"] = 0.0

            # Revision trend (compare estimates over time)
            if len(estimates) >= 2:
                prev = estimates[1]
                if latest.estimated_eps_avg and prev.estimated_eps_avg:
                    features["analyst_revision_trend"] = self._calc_growth(
                        latest.estimated_eps_avg, prev.estimated_eps_avg
                    )
                else:
                    features["analyst_revision_trend"] = 0.0
            else:
                features["analyst_revision_trend"] = 0.0
        else:
            features["analyst_count_eps"] = 0.0
            features["analyst_count_revenue"] = 0.0
            features["eps_estimate_range"] = 0.0
            features["analyst_revision_trend"] = 0.0

        # Price target
        if price_target and current_price and current_price > 0:
            if price_target.target_consensus:
                features["price_target_upside"] = (
                    (price_target.target_consensus - current_price) / current_price
                )
            else:
                features["price_target_upside"] = 0.0

            # Target range (uncertainty)
            if price_target.target_high and price_target.target_low:
                features["price_target_range"] = (
                    (price_target.target_high - price_target.target_low) / current_price
                )
            else:
                features["price_target_range"] = 0.0
        else:
            features["price_target_upside"] = 0.0
            features["price_target_range"] = 0.0

        return features

    def extract_insider_features(
        self,
        transactions: List[InsiderTransaction],
        days_back: int = 90
    ) -> Dict[str, float]:
        """
        Extract features from insider trading data.

        Args:
            transactions: List of InsiderTransaction objects
            days_back: Days to consider for transactions

        Returns:
            Dict of insider features
        """
        features = {}

        if not transactions:
            return self._empty_insider_features()

        # Filter to recent transactions
        # (In production, filter by date - here we assume all are recent)

        purchases = [t for t in transactions if t.is_purchase]
        sales = [t for t in transactions if t.is_sale]

        total_transactions = len(purchases) + len(sales)

        if total_transactions > 0:
            # Insider sentiment (-1 to 1)
            features["insider_sentiment"] = (len(purchases) - len(sales)) / total_transactions
        else:
            features["insider_sentiment"] = 0.0

        # Purchase value
        purchase_value = sum(
            t.transaction_value or 0
            for t in purchases
            if t.transaction_value
        )
        features["insider_buy_value"] = purchase_value

        # Sale value
        sale_value = sum(
            t.transaction_value or 0
            for t in sales
            if t.transaction_value
        )
        features["insider_sell_value"] = sale_value

        # Net value
        features["insider_net_value"] = purchase_value - sale_value

        # Transaction count
        features["insider_transaction_count"] = float(total_transactions)

        return features

    def _empty_insider_features(self) -> Dict[str, float]:
        """Return empty insider features."""
        return {
            "insider_sentiment": 0.0,
            "insider_buy_value": 0.0,
            "insider_sell_value": 0.0,
            "insider_net_value": 0.0,
            "insider_transaction_count": 0.0,
        }

    def extract_institutional_features(
        self,
        holders: List[InstitutionalHolder]
    ) -> Dict[str, float]:
        """
        Extract features from institutional holdings data.

        Args:
            holders: List of InstitutionalHolder objects

        Returns:
            Dict of institutional features
        """
        features = {}

        if not holders:
            return self._empty_institutional_features()

        # Total shares held by institutions
        total_shares = sum(h.shares for h in holders)
        features["institutional_holders_count"] = float(len(holders))
        features["institutional_total_shares"] = float(total_shares)

        # Average change
        changes = [h.change for h in holders if h.change is not None]
        if changes:
            features["institutional_avg_change"] = sum(changes) / len(changes)
            features["institutional_net_change"] = float(sum(changes))

            # Positive vs negative changes
            positive_changes = sum(1 for c in changes if c > 0)
            negative_changes = sum(1 for c in changes if c < 0)
            total = positive_changes + negative_changes
            if total > 0:
                features["institutional_change_sentiment"] = (positive_changes - negative_changes) / total
            else:
                features["institutional_change_sentiment"] = 0.0
        else:
            features["institutional_avg_change"] = 0.0
            features["institutional_net_change"] = 0.0
            features["institutional_change_sentiment"] = 0.0

        return features

    def _empty_institutional_features(self) -> Dict[str, float]:
        """Return empty institutional features."""
        return {
            "institutional_holders_count": 0.0,
            "institutional_total_shares": 0.0,
            "institutional_avg_change": 0.0,
            "institutional_net_change": 0.0,
            "institutional_change_sentiment": 0.0,
        }

    def extract_all(
        self,
        earnings: Optional[EarningsData] = None,
        statements: Optional[List[FinancialStatement]] = None,
        estimates: Optional[List[AnalystEstimate]] = None,
        price_target: Optional[PriceTarget] = None,
        insiders: Optional[List[InsiderTransaction]] = None,
        institutions: Optional[List[InstitutionalHolder]] = None,
        current_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract all financial features.

        Args:
            earnings: EarningsData object
            statements: List of FinancialStatement objects
            estimates: List of AnalystEstimate objects
            price_target: PriceTarget object
            insiders: List of InsiderTransaction objects
            institutions: List of InstitutionalHolder objects
            current_price: Current stock price

        Returns:
            Dict of all financial features with prefix
        """
        features = {}

        # Earnings features
        earnings_features = self.extract_earnings_features(earnings)
        for k, v in earnings_features.items():
            features[f"{self.feature_prefix}{k}"] = v

        # Statement features
        statement_features = self.extract_statement_features(statements or [])
        for k, v in statement_features.items():
            features[f"{self.feature_prefix}{k}"] = v

        # Analyst features
        analyst_features = self.extract_analyst_features(
            estimates or [], price_target, current_price
        )
        for k, v in analyst_features.items():
            features[f"{self.feature_prefix}{k}"] = v

        # Insider features
        insider_features = self.extract_insider_features(insiders or [])
        for k, v in insider_features.items():
            features[f"{self.feature_prefix}{k}"] = v

        # Institutional features
        institutional_features = self.extract_institutional_features(institutions or [])
        for k, v in institutional_features.items():
            features[f"{self.feature_prefix}{k}"] = v

        logger.debug(f"Extracted {len(features)} financial features")
        return features

    def calculate_financial_score(self, features: Dict[str, float]) -> float:
        """
        Calculate overall financial health score (0-10).

        Args:
            features: Dict of financial features

        Returns:
            Score from 0 to 10
        """
        score = 5.0  # Start at neutral

        # Earnings impact (+/- 2 points)
        if features.get(f"{self.feature_prefix}double_beat", 0):
            score += 2.0
        elif features.get(f"{self.feature_prefix}eps_beat", 0):
            score += 1.0
        elif features.get(f"{self.feature_prefix}eps_surprise", 0) < -0.1:
            score -= 2.0

        # Growth impact (+/- 1.5 points)
        yoy_growth = features.get(f"{self.feature_prefix}yoy_revenue_growth", 0)
        if yoy_growth > 0.2:
            score += 1.5
        elif yoy_growth > 0.1:
            score += 0.75
        elif yoy_growth < -0.1:
            score -= 1.5

        # Analyst sentiment (+/- 1 point)
        upside = features.get(f"{self.feature_prefix}price_target_upside", 0)
        if upside > 0.2:
            score += 1.0
        elif upside < -0.1:
            score -= 1.0

        # Insider sentiment (+/- 0.5 points)
        insider = features.get(f"{self.feature_prefix}insider_sentiment", 0)
        if insider > 0.3:
            score += 0.5
        elif insider < -0.3:
            score -= 0.5

        # Clamp to 0-10
        return max(0.0, min(10.0, score))
