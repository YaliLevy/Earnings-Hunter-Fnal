"""
Research insight generator using LLM.

Generates comprehensive research insights from analysis results:
- Executive Summary
- Key Highlights
- Short-term Outlook (1-3 months)
- Long-term Outlook (6-12 months)
- Key Risks
- Bottom Line
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger
from config.settings import Constants

logger = get_logger(__name__)


@dataclass
class ResearchInsight:
    """Research insight structure."""
    symbol: str
    executive_summary: str
    key_highlights: list[str]
    short_term_outlook: str
    long_term_outlook: str
    key_risks: list[str]
    bottom_line: str
    generated_at: str
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class InsightGenerator:
    """
    Generate research insights from analysis results.

    Can use OpenAI API for LLM-powered insights, or
    fall back to rule-based generation.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize insight generator.

        Args:
            openai_api_key: Optional OpenAI API key for LLM generation
        """
        self.openai_api_key = openai_api_key
        self.use_llm = openai_api_key is not None

        if self.use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_api_key)
                logger.info("InsightGenerator initialized with OpenAI")
            except ImportError:
                logger.warning("OpenAI not available, using rule-based generation")
                self.use_llm = False

    def generate(
        self,
        symbol: str,
        financial_summary: Dict[str, Any],
        ceo_summary: Dict[str, Any],
        social_summary: Dict[str, Any],
        prediction: str,
        confidence: Optional[float],
        golden_triangle: Dict[str, Any]
    ) -> ResearchInsight:
        """
        Generate research insight from analysis components.

        Args:
            symbol: Stock ticker
            financial_summary: Financial analysis summary
            ceo_summary: CEO tone analysis summary
            social_summary: Social sentiment summary
            prediction: ML model prediction
            confidence: Prediction confidence
            golden_triangle: Golden Triangle scores

        Returns:
            ResearchInsight object
        """
        from datetime import datetime

        if self.use_llm:
            try:
                return self._generate_with_llm(
                    symbol, financial_summary, ceo_summary,
                    social_summary, prediction, confidence, golden_triangle
                )
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, using rule-based")

        # Fall back to rule-based generation
        return self._generate_rule_based(
            symbol, financial_summary, ceo_summary,
            social_summary, prediction, confidence, golden_triangle
        )

    def _generate_with_llm(
        self,
        symbol: str,
        financial_summary: Dict[str, Any],
        ceo_summary: Dict[str, Any],
        social_summary: Dict[str, Any],
        prediction: str,
        confidence: Optional[float],
        golden_triangle: Dict[str, Any]
    ) -> ResearchInsight:
        """Generate insight using OpenAI LLM."""
        from datetime import datetime

        prompt = self._build_llm_prompt(
            symbol, financial_summary, ceo_summary,
            social_summary, prediction, confidence, golden_triangle
        )

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial analyst providing "
                        "educational research insights. Be specific with numbers "
                        "and data. Always emphasize this is NOT financial advice."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        content = response.choices[0].message.content

        # Parse LLM response
        insight = self._parse_llm_response(content, symbol)

        return insight

    def _build_llm_prompt(
        self,
        symbol: str,
        financial_summary: Dict[str, Any],
        ceo_summary: Dict[str, Any],
        social_summary: Dict[str, Any],
        prediction: str,
        confidence: Optional[float],
        golden_triangle: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM."""
        return f"""
Analyze the following earnings data for {symbol} and generate a research insight.

## Financial Data (40% weight):
- EPS: Actual {financial_summary.get('eps_actual')} vs Est {financial_summary.get('eps_estimated')}
- EPS Surprise: {financial_summary.get('eps_surprise_pct', 'N/A')}%
- Revenue Beat: {financial_summary.get('revenue_beat')}
- Double Beat: {financial_summary.get('double_beat')}

## CEO Tone Analysis (35% weight):
- Tone Summary: {ceo_summary.get('tone_summary', 'N/A')}
- Confidence Score: {ceo_summary.get('confidence_score', 'N/A')}
- Raised Guidance: {ceo_summary.get('raised_guidance', 'N/A')}
- Key Positive: {ceo_summary.get('key_positive_phrases', [])}
- Key Negative: {ceo_summary.get('key_negative_phrases', [])}

## Social Sentiment (25% weight):
- Sentiment: {social_summary.get('sentiment_label', 'N/A')}
- Hype Index: {social_summary.get('hype_index', 'N/A')}
- Post Count: {social_summary.get('post_count', 0)}

## Golden Triangle Scores:
- Financial: {golden_triangle.get('financial', {}).get('score', 'N/A')}/10
- CEO Tone: {golden_triangle.get('ceo_tone', {}).get('score', 'N/A')}/10
- Social: {golden_triangle.get('social', {}).get('score', 'N/A')}/10
- Composite: {golden_triangle.get('composite', {}).get('score', 'N/A')}/10

## ML Prediction: {prediction} (Confidence: {confidence or 'N/A'})

Generate a research insight with these exact sections:
1. EXECUTIVE_SUMMARY: 2-3 sentences on what happened
2. KEY_HIGHLIGHTS: 3-5 bullet points
3. SHORT_TERM_OUTLOOK: 1-3 months outlook
4. LONG_TERM_OUTLOOK: 6-12 months outlook
5. KEY_RISKS: 3-5 risk factors
6. BOTTOM_LINE: One paragraph synthesis

Format each section with its header on a new line followed by the content.
Remember: This is for educational purposes only, NOT financial advice.
"""

    def _parse_llm_response(self, content: str, symbol: str) -> ResearchInsight:
        """Parse LLM response into ResearchInsight."""
        from datetime import datetime

        sections = {
            "executive_summary": "",
            "key_highlights": [],
            "short_term_outlook": "",
            "long_term_outlook": "",
            "key_risks": [],
            "bottom_line": ""
        }

        current_section = None
        current_content = []

        for line in content.split("\n"):
            line = line.strip()
            upper = line.upper()

            if "EXECUTIVE_SUMMARY" in upper:
                current_section = "executive_summary"
                current_content = []
            elif "KEY_HIGHLIGHTS" in upper:
                if current_section and current_content:
                    sections[current_section] = self._join_content(current_section, current_content)
                current_section = "key_highlights"
                current_content = []
            elif "SHORT_TERM" in upper or "SHORT-TERM" in upper:
                if current_section and current_content:
                    sections[current_section] = self._join_content(current_section, current_content)
                current_section = "short_term_outlook"
                current_content = []
            elif "LONG_TERM" in upper or "LONG-TERM" in upper:
                if current_section and current_content:
                    sections[current_section] = self._join_content(current_section, current_content)
                current_section = "long_term_outlook"
                current_content = []
            elif "KEY_RISKS" in upper or "RISKS" in upper:
                if current_section and current_content:
                    sections[current_section] = self._join_content(current_section, current_content)
                current_section = "key_risks"
                current_content = []
            elif "BOTTOM_LINE" in upper or "BOTTOM LINE" in upper:
                if current_section and current_content:
                    sections[current_section] = self._join_content(current_section, current_content)
                current_section = "bottom_line"
                current_content = []
            elif line and current_section:
                current_content.append(line)

        # Handle last section
        if current_section and current_content:
            sections[current_section] = self._join_content(current_section, current_content)

        return ResearchInsight(
            symbol=symbol,
            executive_summary=sections["executive_summary"],
            key_highlights=sections["key_highlights"] if isinstance(sections["key_highlights"], list) else [sections["key_highlights"]],
            short_term_outlook=sections["short_term_outlook"],
            long_term_outlook=sections["long_term_outlook"],
            key_risks=sections["key_risks"] if isinstance(sections["key_risks"], list) else [sections["key_risks"]],
            bottom_line=sections["bottom_line"],
            generated_at=datetime.now().isoformat(),
            disclaimer=Constants.DISCLAIMER_BANNER
        )

    def _join_content(self, section: str, content: list) -> Any:
        """Join content based on section type."""
        if section in ["key_highlights", "key_risks"]:
            # Return as list, clean bullet points
            return [
                line.lstrip("- •*").strip()
                for line in content
                if line.strip()
            ]
        else:
            return " ".join(content)

    def _generate_rule_based(
        self,
        symbol: str,
        financial_summary: Dict[str, Any],
        ceo_summary: Dict[str, Any],
        social_summary: Dict[str, Any],
        prediction: str,
        confidence: Optional[float],
        golden_triangle: Dict[str, Any]
    ) -> ResearchInsight:
        """Generate insight using rule-based logic."""
        from datetime import datetime

        # Executive Summary
        exec_summary = self._generate_executive_summary(
            symbol, financial_summary, ceo_summary, prediction
        )

        # Key Highlights
        highlights = self._generate_highlights(financial_summary, ceo_summary, social_summary)

        # Short-term Outlook
        short_term = self._generate_short_term(prediction, confidence, golden_triangle)

        # Long-term Outlook
        long_term = self._generate_long_term(financial_summary, ceo_summary)

        # Key Risks
        risks = self._generate_risks(financial_summary, ceo_summary, social_summary)

        # Bottom Line
        bottom_line = self._generate_bottom_line(
            symbol, prediction, confidence, golden_triangle
        )

        return ResearchInsight(
            symbol=symbol,
            executive_summary=exec_summary,
            key_highlights=highlights,
            short_term_outlook=short_term,
            long_term_outlook=long_term,
            key_risks=risks,
            bottom_line=bottom_line,
            generated_at=datetime.now().isoformat(),
            disclaimer=Constants.DISCLAIMER_BANNER
        )

    def _generate_executive_summary(
        self,
        symbol: str,
        financial: Dict[str, Any],
        ceo: Dict[str, Any],
        prediction: str
    ) -> str:
        """Generate executive summary."""
        parts = []

        # Earnings result
        if financial.get("double_beat"):
            parts.append(f"{symbol} delivered a strong quarter, beating both EPS and revenue estimates.")
        elif financial.get("eps_beat"):
            parts.append(f"{symbol} beat EPS estimates but missed on revenue.")
        elif financial.get("revenue_beat"):
            parts.append(f"{symbol} beat revenue estimates but missed on EPS.")
        else:
            parts.append(f"{symbol} missed analyst estimates this quarter.")

        # CEO tone
        tone = ceo.get("tone_summary", "Unknown")
        if tone == "Confident":
            parts.append("Management expressed strong confidence in the business outlook.")
        elif tone == "Cautious":
            parts.append("Management took a cautious tone regarding near-term expectations.")
        elif tone == "Defensive":
            parts.append("Management appeared defensive during the earnings call.")

        # Prediction
        if prediction == "Growth":
            parts.append("Our model suggests potential upside in the near term.")
        elif prediction == "Risk":
            parts.append("Our model flags elevated risk for the near term.")

        return " ".join(parts)

    def _generate_highlights(
        self,
        financial: Dict[str, Any],
        ceo: Dict[str, Any],
        social: Dict[str, Any]
    ) -> list[str]:
        """Generate key highlights."""
        highlights = []

        # Financial highlights
        eps_surprise = financial.get("eps_surprise_pct")
        if eps_surprise:
            direction = "beat" if eps_surprise > 0 else "missed"
            highlights.append(f"EPS {direction} estimates by {abs(eps_surprise):.1f}%")

        rev_surprise = financial.get("revenue_surprise_pct")
        if rev_surprise:
            direction = "beat" if rev_surprise > 0 else "missed"
            highlights.append(f"Revenue {direction} estimates by {abs(rev_surprise):.1f}%")

        # CEO highlights
        if ceo.get("raised_guidance"):
            highlights.append("Management raised forward guidance")
        elif ceo.get("lowered_guidance"):
            highlights.append("Management lowered forward guidance")

        tone = ceo.get("tone_summary")
        if tone:
            highlights.append(f"CEO tone assessed as: {tone}")

        # Social highlights
        if social.get("available"):
            hype = social.get("hype_level", "Unknown")
            sentiment = social.get("sentiment_label", "Neutral")
            highlights.append(f"Social sentiment: {sentiment} with {hype.lower()} retail interest")

        return highlights[:5]  # Max 5 highlights

    def _generate_short_term(
        self,
        prediction: str,
        confidence: Optional[float],
        golden_triangle: Dict[str, Any]
    ) -> str:
        """Generate short-term outlook."""
        composite = golden_triangle.get("composite", {}).get("score", 5)

        if prediction == "Growth" and composite > 6:
            outlook = (
                "The near-term outlook appears favorable based on strong fundamentals "
                "and positive sentiment indicators. Watch for potential continuation "
                "of momentum in the coming weeks."
            )
        elif prediction == "Risk" and composite < 4:
            outlook = (
                "Caution is warranted in the near term. The combination of earnings "
                "results and sentiment suggests potential headwinds. Monitor for "
                "stabilization before considering positions."
            )
        else:
            outlook = (
                "The near-term picture is mixed. While some indicators are positive, "
                "others suggest uncertainty. A wait-and-see approach may be prudent "
                "until clearer signals emerge."
            )

        if confidence:
            conf_pct = confidence * 100
            outlook += f" Model confidence: {conf_pct:.0f}%."

        return outlook

    def _generate_long_term(
        self,
        financial: Dict[str, Any],
        ceo: Dict[str, Any]
    ) -> str:
        """Generate long-term outlook."""
        parts = []

        # Based on guidance
        if ceo.get("raised_guidance"):
            parts.append(
                "Management's raised guidance suggests confidence in sustainable growth. "
                "This could support a positive trajectory over the coming quarters."
            )
        elif ceo.get("lowered_guidance"):
            parts.append(
                "The lowered guidance indicates management sees challenges ahead. "
                "Recovery may take several quarters depending on market conditions."
            )
        else:
            parts.append(
                "With guidance maintained, the long-term outlook depends on execution "
                "and broader market conditions."
            )

        # Based on CEO tone
        tone = ceo.get("tone_summary")
        if tone == "Confident":
            parts.append(
                "Management's confident tone suggests they see a clear path to growth."
            )
        elif tone == "Cautious":
            parts.append(
                "The cautious management tone suggests some uncertainty in the outlook."
            )

        return " ".join(parts)

    def _generate_risks(
        self,
        financial: Dict[str, Any],
        ceo: Dict[str, Any],
        social: Dict[str, Any]
    ) -> list[str]:
        """Generate key risks."""
        risks = []

        # Earnings miss risk
        if not financial.get("eps_beat"):
            risks.append("Recent earnings miss may pressure analyst estimates")

        # Guidance risk
        if ceo.get("lowered_guidance"):
            risks.append("Lowered guidance indicates management sees headwinds")

        # CEO uncertainty
        uncertainty = ceo.get("uncertainty_ratio", 0)
        if uncertainty > 0.1:
            risks.append("High uncertainty language from management during earnings call")

        # Social risk
        if social.get("available"):
            wsb = social.get("wsb_ratio", 0)
            if wsb > 0.5:
                risks.append("High speculative retail interest may increase volatility")

            sentiment = social.get("sentiment_score", 0)
            if sentiment < -0.2:
                risks.append("Negative retail sentiment could pressure near-term performance")

        # General risks
        risks.append("Market conditions and macroeconomic factors")
        risks.append("Execution risk on stated initiatives")

        return risks[:5]  # Max 5 risks

    def _generate_bottom_line(
        self,
        symbol: str,
        prediction: str,
        confidence: Optional[float],
        golden_triangle: Dict[str, Any]
    ) -> str:
        """Generate bottom line summary."""
        fin_score = golden_triangle.get("financial", {}).get("score", 5)
        ceo_score = golden_triangle.get("ceo_tone", {}).get("score", 5)
        social_score = golden_triangle.get("social", {}).get("score", 5)
        composite = golden_triangle.get("composite", {}).get("score", 5)

        if prediction == "Growth" and composite > 6:
            assessment = (
                f"{symbol}'s latest earnings present a positive picture across our "
                f"Golden Triangle analysis. Financial metrics score {fin_score:.1f}/10, "
                f"CEO tone {ceo_score:.1f}/10, and social sentiment {social_score:.1f}/10, "
                f"combining for a weighted score of {composite:.1f}/10. "
                "The alignment of fundamentals with positive sentiment suggests "
                "favorable conditions for the stock."
            )
        elif prediction == "Risk" and composite < 4:
            assessment = (
                f"{symbol} faces challenges based on our analysis. "
                f"The Golden Triangle scores (Financial: {fin_score:.1f}, "
                f"CEO Tone: {ceo_score:.1f}, Social: {social_score:.1f}) "
                f"yield a composite of {composite:.1f}/10. "
                "Investors should exercise caution and monitor for improvement "
                "in key metrics before considering exposure."
            )
        else:
            assessment = (
                f"{symbol} presents a mixed picture. Our Golden Triangle analysis "
                f"shows Financial: {fin_score:.1f}/10, CEO Tone: {ceo_score:.1f}/10, "
                f"Social: {social_score:.1f}/10, for a composite of {composite:.1f}/10. "
                "The divergence in signals suggests waiting for clearer direction "
                "may be prudent."
            )

        assessment += (
            " Remember: This analysis is for educational purposes only and should "
            "not be considered financial advice."
        )

        return assessment
