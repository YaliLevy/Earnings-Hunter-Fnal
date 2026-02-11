"""
AI-powered news filtering and sentiment analysis.

Uses GPT-4o-mini to filter relevant news and classify sentiment.
"""

import os
from typing import List, Dict, Any
from openai import OpenAI

from src.utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Export OpenAI API key
os.environ["OPENAI_API_KEY"] = settings.openai_api_key


class NewsAnalyzer:
    """
    AI-powered news analyzer for stock-specific sentiment analysis.

    Uses GPT-4o-mini for:
    1. Filtering relevant news (stock-specific, not market-general)
    2. Classifying sentiment (Bullish/Bearish/Neutral)
    """

    def __init__(self):
        """Initialize the news analyzer with OpenAI client."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"

    def analyze_news_batch(
        self,
        symbol: str,
        company_name: str,
        news_articles: List[Dict[str, Any]],
        max_results: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Analyze a batch of news articles using AI.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            company_name: Company name (e.g., "Apple Inc.")
            news_articles: List of news articles with 'title' and 'text'
            max_results: Maximum number of relevant articles to return

        Returns:
            List of filtered and analyzed articles with AI sentiment
        """
        if not news_articles:
            logger.warning(f"No news articles provided for {symbol}")
            return []

        try:
            # Prepare articles for AI analysis
            articles_text = ""
            for i, article in enumerate(news_articles[:30], 1):  # Limit to 30 for token efficiency
                title = article.get('title', '')
                text = article.get('text', '')[:200]  # First 200 chars
                articles_text += f"\n{i}. {title}\n   {text}\n"

            # AI prompt for filtering and sentiment analysis
            prompt = f"""You are a financial news analyst. Analyze these news articles about {company_name} ({symbol}).

Your tasks:
1. **Filter**: Select only articles DIRECTLY about {company_name}'s business, earnings, products, or stock performance.
   - EXCLUDE: General market news, sector trends, competitor news (unless comparing to {symbol})
   - INCLUDE: Company-specific news, earnings reports, product launches, legal issues, management changes

2. **Classify Sentiment**: For each relevant article, classify as:
   - **Bullish** (+1): Positive news (earnings beat, growth, positive guidance, partnerships)
   - **Bearish** (-1): Negative news (earnings miss, scandals, downgrades, legal troubles)
   - **Neutral** (0): Factual news without clear positive/negative impact

Articles:
{articles_text}

Respond ONLY with a JSON array (no markdown, no explanation):
[
  {{"id": 1, "relevant": true, "sentiment": "bullish", "score": 0.8, "reason": "brief reason"}},
  {{"id": 3, "relevant": true, "sentiment": "bearish", "score": -0.6, "reason": "brief reason"}},
  ...
]

Rules:
- Include ONLY relevant articles (relevant: true)
- Score: -1.0 to +1.0 (magnitude of impact)
- Limit to top {max_results} most relevant articles
"""

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial news analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            # Parse AI response
            ai_response = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]

            import json
            ai_results = json.loads(ai_response)

            # Map AI results back to original articles
            filtered_articles = []
            for ai_result in ai_results:
                article_id = ai_result.get("id", 0)
                if 1 <= article_id <= len(news_articles):
                    original_article = news_articles[article_id - 1]

                    filtered_articles.append({
                        "title": original_article.get("title", ""),
                        "text": original_article.get("text", ""),
                        "url": original_article.get("url"),
                        "source": original_article.get("source", "Unknown"),
                        "date": original_article.get("date"),
                        "sentiment": ai_result.get("sentiment", "neutral"),
                        "score": ai_result.get("score", 0.0),
                        "ai_reason": ai_result.get("reason", "")
                    })

            logger.info(f"AI filtered {len(news_articles)} articles → {len(filtered_articles)} relevant for {symbol}")
            return filtered_articles

        except Exception as e:
            logger.error(f"Error in AI news analysis for {symbol}: {e}")
            # Fallback: return original articles with basic sentiment
            return self._fallback_analysis(news_articles, max_results)

    def _fallback_analysis(
        self,
        news_articles: List[Dict[str, Any]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback sentiment analysis using keyword matching.

        Used when AI analysis fails.
        """
        logger.warning("Using fallback keyword-based sentiment analysis")

        positive_keywords = ['beat', 'growth', 'surge', 'record', 'strong', 'upgrade', 'profit', 'buy']
        negative_keywords = ['miss', 'loss', 'decline', 'drop', 'weak', 'downgrade', 'concern', 'sell']

        results = []
        for article in news_articles[:max_results]:
            title_lower = article.get('title', '').lower()
            text_lower = article.get('text', '').lower()
            combined = title_lower + ' ' + text_lower

            positive_count = sum(1 for kw in positive_keywords if kw in combined)
            negative_count = sum(1 for kw in negative_keywords if kw in combined)

            if positive_count > negative_count:
                sentiment = "bullish"
                score = min(0.8, positive_count * 0.2)
            elif negative_count > positive_count:
                sentiment = "bearish"
                score = max(-0.8, -negative_count * 0.2)
            else:
                sentiment = "neutral"
                score = 0.0

            results.append({
                "title": article.get("title", ""),
                "text": article.get("text", ""),
                "url": article.get("url"),
                "source": article.get("source", "Unknown"),
                "date": article.get("date"),
                "sentiment": sentiment,
                "score": score,
                "ai_reason": "Fallback keyword analysis"
            })

        return results


# Singleton instance
_news_analyzer = None


def get_news_analyzer() -> NewsAnalyzer:
    """Get or create NewsAnalyzer singleton."""
    global _news_analyzer
    if _news_analyzer is None:
        _news_analyzer = NewsAnalyzer()
    return _news_analyzer
