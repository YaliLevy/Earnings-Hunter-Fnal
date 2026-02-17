"""
Financial Expert Scorer - AI-powered scoring using GPT-4o-mini.

A lightweight agent that receives all raw data and returns
structured scores (0-100) for each analysis component.
"""

import json
import os
from typing import Dict, Any, Optional, List

from openai import OpenAI

from src.utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)

# Default neutral scores when API fails
DEFAULT_SCORES = {
    "financial_score": 50,
    "financial_reasoning": "Unable to score - using neutral default",
    "ceo_tone_score": 50,
    "ceo_tone_reasoning": "Unable to score - using neutral default",
    "news_score": 50,
    "news_reasoning": "Unable to score - using neutral default",
    "insider_score": 50,
    "insider_reasoning": "Unable to score - using neutral default",
    "overall_reasoning": "Scoring unavailable - defaults applied",
}

SYSTEM_PROMPT = """You are a senior financial analyst with 20 years of experience analyzing earnings reports.
You specialize in post-earnings price movement prediction.

Your task: Score each data category from 0 to 100 where:
- 0-20: Very bearish signal
- 20-40: Bearish signal
- 40-60: Neutral / mixed signals
- 60-80: Bullish signal
- 80-100: Very bullish signal

Be specific in your reasoning. Reference actual numbers from the data.
Return ONLY valid JSON, no markdown or extra text."""


class FinancialExpertScorer:
    """Scores analysis components using GPT-4o-mini."""

    def __init__(self):
        settings = get_settings()
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"

    def score(
        self,
        financial_summary: Dict[str, Any],
        transcript_content: Optional[str],
        news_articles: Optional[List[Dict]],
        insider_transactions: Optional[List[Dict]],
        prediction_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Score all components using GPT-4o-mini.

        Returns dict with scores (0-100) and reasoning for each component.
        Falls back to neutral scores (50) on any failure.
        """
        try:
            prompt = self._build_prompt(
                financial_summary,
                transcript_content,
                news_articles,
                insider_transactions,
                prediction_result,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=15,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            scores = json.loads(raw)
            scores = self._validate_scores(scores)

            logger.info(
                f"Expert scores: fin={scores['financial_score']}, "
                f"ceo={scores['ceo_tone_score']}, news={scores['news_score']}, "
                f"insider={scores['insider_score']}"
            )
            return scores

        except Exception as e:
            logger.warning(f"Financial expert scoring failed: {e}. Using defaults.")
            return DEFAULT_SCORES.copy()

    def _build_prompt(
        self,
        fin: Dict[str, Any],
        transcript: Optional[str],
        news: Optional[List[Dict]],
        insiders: Optional[List[Dict]],
        prediction: Optional[Dict[str, Any]],
    ) -> str:
        """Build the scoring prompt with all available data."""

        sections = []

        # Financial data
        eps_actual = fin.get("eps_actual")
        eps_est = fin.get("eps_estimated")
        eps_surprise = fin.get("eps_surprise_pct") or fin.get("eps_surprise")
        rev_actual = fin.get("revenue_actual")
        rev_est = fin.get("revenue_estimated")
        rev_surprise = fin.get("revenue_surprise_pct") or fin.get("revenue_surprise")

        sections.append(f"""FINANCIAL DATA:
- EPS: actual ${eps_actual} vs estimated ${eps_est} (surprise: {eps_surprise:.1f}% {'beat' if fin.get('eps_beat') else 'miss'})
- Revenue: actual ${self._fmt_large(rev_actual)} vs estimated ${self._fmt_large(rev_est)} (surprise: {rev_surprise:.1f}% {'beat' if fin.get('revenue_beat') else 'miss'})
- Double beat (EPS + Revenue): {'Yes' if fin.get('double_beat') else 'No'}
- Current price: ${fin.get('current_price', 'N/A')}
- Market cap: ${self._fmt_large(fin.get('market_cap'))}""")

        # CEO Tone
        if transcript:
            truncated = transcript[:3000]
            sections.append(f"""CEO TONE (earnings call transcript excerpt):
{truncated}""")
        else:
            sections.append("CEO TONE: No earnings call transcript available. Score as neutral (50).")

        # News
        if news:
            bullish = sum(1 for a in news if a.get("sentiment") == "bullish")
            bearish = sum(1 for a in news if a.get("sentiment") == "bearish")
            neutral = sum(1 for a in news if a.get("sentiment") == "neutral")
            top_headlines = "\n".join(
                f"  - [{a.get('sentiment', 'neutral').upper()}] {a.get('title', '')}"
                for a in news[:8]
            )
            sections.append(f"""NEWS SENTIMENT:
- {len(news)} articles analyzed: {bullish} bullish, {bearish} bearish, {neutral} neutral
- Top headlines:
{top_headlines}""")
        else:
            sections.append("NEWS SENTIMENT: No news articles available. Score as neutral (50).")

        # Insider activity
        if insiders:
            buys = [t for t in insiders if t.get("type") == "BUY"]
            sells = [t for t in insiders if t.get("type") == "SELL"]
            buy_value = sum(t.get("value", 0) or 0 for t in buys)
            sell_value = sum(t.get("value", 0) or 0 for t in sells)
            sections.append(f"""INSIDER ACTIVITY:
- Buy transactions: {len(buys)} (total value: ${self._fmt_large(buy_value)})
- Sell transactions: {len(sells)} (total value: ${self._fmt_large(sell_value)})
- Net direction: {'More buying' if len(buys) > len(sells) else 'More selling' if len(sells) > len(buys) else 'Balanced'}""")
        else:
            sections.append("INSIDER ACTIVITY: No insider transaction data available. Score as neutral (50).")

        # ML context
        if prediction:
            sections.append(f"""ML MODEL CONTEXT (for reference only, do not score this):
- Prediction: {prediction.get('consensus_prediction', 'Unknown')}
- Best model confidence: {prediction.get('best_model_confidence', 'N/A')}
- Models agreeing: {prediction.get('models_agree', 'N/A')}/{prediction.get('models_total', 'N/A')}""")

        prompt = "\n\n".join(sections)
        prompt += """

Score each category 0-100. Return ONLY this JSON:
{
  "financial_score": <0-100>,
  "financial_reasoning": "<1 sentence>",
  "ceo_tone_score": <0-100>,
  "ceo_tone_reasoning": "<1 sentence>",
  "news_score": <0-100>,
  "news_reasoning": "<1 sentence>",
  "insider_score": <0-100>,
  "insider_reasoning": "<1 sentence>",
  "overall_reasoning": "<2-3 sentence synthesis>"
}"""

        return prompt

    def _validate_scores(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp scores to 0-100 range."""
        for key in ["financial_score", "ceo_tone_score", "news_score", "insider_score"]:
            if key in scores:
                try:
                    scores[key] = max(0, min(100, int(scores[key])))
                except (ValueError, TypeError):
                    scores[key] = 50

            # Ensure reasoning exists
            reasoning_key = key.replace("_score", "_reasoning")
            if reasoning_key not in scores or not scores[reasoning_key]:
                scores[reasoning_key] = "No reasoning provided"

        if "overall_reasoning" not in scores:
            scores["overall_reasoning"] = "No overall reasoning provided"

        return scores

    @staticmethod
    def _fmt_large(value) -> str:
        """Format large numbers for readability."""
        if value is None:
            return "N/A"
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)

        if abs(value) >= 1e12:
            return f"{value/1e12:.1f}T"
        if abs(value) >= 1e9:
            return f"{value/1e9:.1f}B"
        if abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        if abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        return f"{value:.2f}"
