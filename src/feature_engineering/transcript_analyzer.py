"""
Earnings call transcript analyzer.

CRITICAL: This module provides 35% of the Golden Triangle (CEO Tone).
Analyzes CEO/CFO language patterns, confidence levels, and forward guidance.
"""

import re
import os
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

from src.utils.logger import get_logger
from src.feature_engineering.sentiment_features import SentimentFeatureExtractor
from config.settings import Constants, get_settings

logger = get_logger(__name__)
settings = get_settings()

# Export OpenAI API key
os.environ["OPENAI_API_KEY"] = settings.openai_api_key


class TranscriptAnalyzer:
    """
    Analyze earnings call transcripts for CEO tone and sentiment.

    This is a CRITICAL component - 35% of the Golden Triangle weighting.

    Features extracted:
    - CEO confidence score
    - Uncertainty ratio
    - Forward guidance sentiment
    - Key positive/negative phrases
    - Overall tone classification
    """

    def __init__(self):
        """Initialize transcript analyzer."""
        self.sentiment_extractor = SentimentFeatureExtractor()

        # Confidence words (positive signals)
        self.confidence_words = [w.lower() for w in Constants.CONFIDENCE_WORDS]

        # Uncertainty words (negative signals)
        self.uncertainty_words = [w.lower() for w in Constants.UNCERTAINTY_WORDS]

        # Forward guidance keywords
        self.guidance_keywords = [w.lower() for w in Constants.GUIDANCE_KEYWORDS]

        # Executive titles to identify speaker sections
        self.executive_titles = [
            "ceo", "chief executive", "president",
            "cfo", "chief financial", "finance officer",
            "coo", "chief operating",
        ]

        # Analyst identifiers to filter out Q&A
        self.analyst_identifiers = [
            "analyst", "question", "q:", "[analyst]",
            "from morgan", "from goldman", "from jpmorgan",
            "from citi", "from wells", "from bank of"
        ]

        self.feature_prefix = "ceo_"

    def extract_ceo_sections(self, transcript: str) -> str:
        """
        Extract CEO and CFO speaking sections from full transcript.

        Filters out analyst questions and other speakers to focus
        on management's actual statements.

        Args:
            transcript: Full earnings call transcript

        Returns:
            Cleaned text containing only executive statements
        """
        if not transcript:
            return ""

        lines = transcript.split('\n')
        executive_sections = []
        is_executive_speaking = False
        current_section = []

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line indicates an executive is speaking
            is_exec_line = any(title in line_lower for title in self.executive_titles)

            # Check if this line indicates an analyst is speaking
            is_analyst_line = any(ident in line_lower for ident in self.analyst_identifiers)

            if is_exec_line and not is_analyst_line:
                # New executive section starting
                if current_section:
                    executive_sections.append(' '.join(current_section))
                    current_section = []
                is_executive_speaking = True

            elif is_analyst_line:
                # Analyst speaking - stop collecting
                if current_section:
                    executive_sections.append(' '.join(current_section))
                    current_section = []
                is_executive_speaking = False

            elif is_executive_speaking:
                # Continue collecting executive speech
                if line.strip():
                    current_section.append(line.strip())

        # Don't forget the last section
        if current_section:
            executive_sections.append(' '.join(current_section))

        # If we couldn't parse sections, return the whole transcript
        # (better than nothing)
        if not executive_sections:
            logger.warning("Could not parse executive sections, using full transcript")
            return transcript

        result = '\n'.join(executive_sections)
        logger.debug(f"Extracted {len(executive_sections)} executive sections")
        return result

    def analyze_confidence_language(self, text: str) -> Dict[str, any]:
        """
        Analyze confidence vs uncertainty word patterns.

        Args:
            text: Text to analyze (preferably CEO/CFO sections only)

        Returns:
            Dict with confidence analysis results
        """
        if not text:
            return self._empty_confidence_result()

        text_lower = text.lower()
        words = re.findall(r'\b[a-z]+\b', text_lower)
        total_words = len(words)

        if total_words == 0:
            return self._empty_confidence_result()

        # Find confidence words
        confidence_found = []
        for word in self.confidence_words:
            count = text_lower.count(word)
            if count > 0:
                confidence_found.extend([word] * count)

        # Find uncertainty words
        uncertainty_found = []
        for word in self.uncertainty_words:
            count = text_lower.count(word)
            if count > 0:
                uncertainty_found.extend([word] * count)

        # Calculate ratios
        confidence_count = len(confidence_found)
        uncertainty_count = len(uncertainty_found)

        # Normalize by total words (per 1000 words)
        confidence_ratio = (confidence_count / total_words) * 1000
        uncertainty_ratio = (uncertainty_count / total_words) * 1000

        # Calculate confidence score (0 to 1)
        # More confidence words relative to uncertainty = higher score
        total_signal_words = confidence_count + uncertainty_count
        if total_signal_words > 0:
            # confidence_score = confidence_count / total_signal_words
            # This gives 0.0 when all uncertainty, 1.0 when all confidence, 0.5 when equal
            confidence_score = confidence_count / total_signal_words
        else:
            # Default to neutral (0.5) when no signal words found
            confidence_score = 0.5

        return {
            "confidence_ratio": confidence_ratio,
            "uncertainty_ratio": uncertainty_ratio,
            "confidence_score": confidence_score,
            "confidence_words_found": list(set(confidence_found)),
            "uncertainty_words_found": list(set(uncertainty_found)),
            "confidence_count": confidence_count,
            "uncertainty_count": uncertainty_count,
            "total_words": total_words
        }

    def _empty_confidence_result(self) -> Dict[str, any]:
        """Return empty confidence analysis result."""
        return {
            "confidence_ratio": 0.0,
            "uncertainty_ratio": 0.0,
            "confidence_score": 0.0,
            "confidence_words_found": [],
            "uncertainty_words_found": [],
            "confidence_count": 0,
            "uncertainty_count": 0,
            "total_words": 0
        }

    def extract_forward_guidance(self, text: str) -> Dict[str, any]:
        """
        Extract and analyze forward-looking statements.

        Identifies statements about future expectations, guidance,
        and outlook to assess management's forward sentiment.

        Args:
            text: Text to analyze

        Returns:
            Dict with forward guidance analysis
        """
        if not text:
            return self._empty_guidance_result()

        text_lower = text.lower()

        # Split into sentences
        sentences = re.split(r'[.!?]', text)

        # Find guidance sentences
        guidance_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(kw in sentence_lower for kw in self.guidance_keywords):
                if len(sentence.strip()) > 20:  # Filter out very short matches
                    guidance_sentences.append(sentence.strip())

        if not guidance_sentences:
            return self._empty_guidance_result()

        # Analyze sentiment of guidance statements
        guidance_sentiments = []
        for sentence in guidance_sentences:
            sentiment = self.sentiment_extractor.analyze_text(sentence)
            guidance_sentiments.append(sentiment["combined_sentiment"])

        avg_guidance_sentiment = sum(guidance_sentiments) / len(guidance_sentiments)

        # Detect raised/lowered guidance
        raised_keywords = ["raise", "raised", "increase", "higher", "above", "exceed", "beat"]
        lowered_keywords = ["lower", "lowered", "reduce", "below", "cut", "revised down"]

        has_raised = any(kw in text_lower for kw in raised_keywords)
        has_lowered = any(kw in text_lower for kw in lowered_keywords)

        return {
            "guidance_statements": guidance_sentences[:5],  # Top 5
            "guidance_count": len(guidance_sentences),
            "guidance_sentiment": avg_guidance_sentiment,
            "has_raised_guidance": has_raised,
            "has_lowered_guidance": has_lowered,
            "guidance_sentiments": guidance_sentiments
        }

    def _empty_guidance_result(self) -> Dict[str, any]:
        """Return empty guidance analysis result."""
        return {
            "guidance_statements": [],
            "guidance_count": 0,
            "guidance_sentiment": 0.0,
            "has_raised_guidance": False,
            "has_lowered_guidance": False,
            "guidance_sentiments": []
        }

    def extract_key_phrases(
        self,
        text: str,
        top_n: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        Extract key positive and negative phrases.

        Args:
            text: Text to analyze
            top_n: Number of phrases to return

        Returns:
            Tuple of (positive_phrases, negative_phrases)
        """
        if not text:
            return [], []

        # Split into sentences
        sentences = re.split(r'[.!?]', text)

        positive_phrases = []
        negative_phrases = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue

            sentiment = self.sentiment_extractor.analyze_text(sentence)
            compound = sentiment["combined_sentiment"]

            if compound >= 0.3:
                positive_phrases.append((sentence, compound))
            elif compound <= -0.3:
                negative_phrases.append((sentence, compound))

        # Sort by sentiment strength
        positive_phrases.sort(key=lambda x: x[1], reverse=True)
        negative_phrases.sort(key=lambda x: x[1])

        return (
            [p[0] for p in positive_phrases[:top_n]],
            [p[0] for p in negative_phrases[:top_n]]
        )

    def classify_tone(
        self,
        confidence_score: float,
        sentiment_score: float,
        guidance_sentiment: float
    ) -> str:
        """
        Classify overall CEO tone into category.

        Args:
            confidence_score: Confidence analysis score (-1 to 1)
            sentiment_score: Overall sentiment score (-1 to 1)
            guidance_sentiment: Forward guidance sentiment (-1 to 1)

        Returns:
            Tone classification string
        """
        # Weighted average
        combined = (
            confidence_score * 0.4 +
            sentiment_score * 0.3 +
            guidance_sentiment * 0.3
        )

        if combined >= 0.3:
            if confidence_score >= 0.4:
                return "Confident"
            else:
                return "Optimistic"
        elif combined >= 0:
            return "Cautious"
        elif combined >= -0.3:
            return "Defensive"
        else:
            return "Pessimistic"

    def calculate_ceo_sentiment_score(
        self,
        transcript: str
    ) -> Dict[str, any]:
        """
        Main function - comprehensive transcript analysis.

        This is the primary entry point for transcript analysis.
        Combines all analysis methods into a single result.

        Args:
            transcript: Full earnings call transcript

        Returns:
            Complete CEO sentiment analysis dict
        """
        if not transcript:
            return self._empty_ceo_result()

        # Step 1: Use full transcript for more comprehensive analysis
        # Note: CEO section extraction often misses important content,
        # and full transcript analysis gives better signal
        ceo_text = transcript
        logger.info(f"Analyzing full transcript ({len(transcript)} chars)")

        # Step 2: Analyze confidence language
        confidence_analysis = self.analyze_confidence_language(ceo_text)

        # Step 3: Analyze overall sentiment
        sentiment_analysis = self.sentiment_extractor.analyze_text(ceo_text)

        # Step 4: Extract forward guidance
        guidance_analysis = self.extract_forward_guidance(ceo_text)

        # Step 5: Extract key phrases
        positive_phrases, negative_phrases = self.extract_key_phrases(ceo_text)

        # Step 6: Classify overall tone
        tone_summary = self.classify_tone(
            confidence_analysis["confidence_score"],
            sentiment_analysis["combined_sentiment"],
            guidance_analysis["guidance_sentiment"]
        )

        # Step 7: Calculate CEO score (0-10)
        ceo_score = self._calculate_ceo_score(
            confidence_analysis["confidence_score"],
            sentiment_analysis["combined_sentiment"],
            guidance_analysis["guidance_sentiment"],
            guidance_analysis["has_raised_guidance"],
            guidance_analysis["has_lowered_guidance"]
        )

        return {
            # Main scores
            "ceo_confidence_score": confidence_analysis["confidence_score"],
            "ceo_overall_sentiment": sentiment_analysis["combined_sentiment"],
            "forward_guidance_sentiment": guidance_analysis["guidance_sentiment"],
            "uncertainty_ratio": confidence_analysis["uncertainty_ratio"],
            "tone_summary": tone_summary,
            "ceo_score": ceo_score,

            # Detailed analysis
            "confidence_analysis": confidence_analysis,
            "sentiment_analysis": sentiment_analysis,
            "guidance_analysis": guidance_analysis,

            # Key phrases
            "key_positive_phrases": positive_phrases,
            "key_negative_phrases": negative_phrases,

            # Metadata
            "total_words_analyzed": confidence_analysis["total_words"],
            "guidance_statement_count": guidance_analysis["guidance_count"],
        }

    def _empty_ceo_result(self) -> Dict[str, any]:
        """Return empty CEO analysis result."""
        return {
            "ceo_confidence_score": 0.0,
            "ceo_overall_sentiment": 0.0,
            "forward_guidance_sentiment": 0.0,
            "uncertainty_ratio": 0.0,
            "tone_summary": "Unknown",
            "ceo_score": 5.0,
            "confidence_analysis": self._empty_confidence_result(),
            "sentiment_analysis": {},
            "guidance_analysis": self._empty_guidance_result(),
            "key_positive_phrases": [],
            "key_negative_phrases": [],
            "total_words_analyzed": 0,
            "guidance_statement_count": 0,
        }

    def _calculate_ceo_score(
        self,
        confidence_score: float,
        sentiment_score: float,
        guidance_sentiment: float,
        raised_guidance: bool,
        lowered_guidance: bool
    ) -> float:
        """
        Calculate CEO tone score (0-10).

        Args:
            confidence_score: -1 to 1
            sentiment_score: -1 to 1
            guidance_sentiment: -1 to 1
            raised_guidance: Whether guidance was raised
            lowered_guidance: Whether guidance was lowered

        Returns:
            Score from 0 to 10
        """
        # Start at neutral
        score = 5.0

        # Confidence impact (+/- 2 points)
        score += confidence_score * 2.0

        # Sentiment impact (+/- 1.5 points)
        score += sentiment_score * 1.5

        # Guidance sentiment impact (+/- 1 point)
        score += guidance_sentiment * 1.0

        # Raised/lowered guidance (+/- 0.5 points)
        if raised_guidance:
            score += 0.5
        if lowered_guidance:
            score -= 0.5

        # Clamp to 0-10
        return max(0.0, min(10.0, score))

    def extract_features(self, transcript: str) -> Dict[str, float]:
        """
        Extract ML-ready features from transcript.

        Returns features with prefix for use in model training.

        Args:
            transcript: Full earnings call transcript

        Returns:
            Dict of numeric features with prefix
        """
        analysis = self.calculate_ceo_sentiment_score(transcript)

        features = {
            f"{self.feature_prefix}confidence_score": analysis["ceo_confidence_score"],
            f"{self.feature_prefix}overall_sentiment": analysis["ceo_overall_sentiment"],
            f"{self.feature_prefix}forward_guidance_sentiment": analysis["forward_guidance_sentiment"],
            f"{self.feature_prefix}uncertainty_ratio": analysis["uncertainty_ratio"],
            f"{self.feature_prefix}score": analysis["ceo_score"],

            # Additional numeric features
            f"{self.feature_prefix}confidence_ratio": analysis["confidence_analysis"]["confidence_ratio"],
            f"{self.feature_prefix}confidence_count": float(analysis["confidence_analysis"]["confidence_count"]),
            f"{self.feature_prefix}uncertainty_count": float(analysis["confidence_analysis"]["uncertainty_count"]),
            f"{self.feature_prefix}guidance_count": float(analysis["guidance_statement_count"]),
            f"{self.feature_prefix}raised_guidance": 1.0 if analysis["guidance_analysis"]["has_raised_guidance"] else 0.0,
            f"{self.feature_prefix}lowered_guidance": 1.0 if analysis["guidance_analysis"]["has_lowered_guidance"] else 0.0,
            f"{self.feature_prefix}total_words": float(analysis["total_words_analyzed"]),

            # Encode tone as numeric
            f"{self.feature_prefix}tone_confident": 1.0 if analysis["tone_summary"] == "Confident" else 0.0,
            f"{self.feature_prefix}tone_optimistic": 1.0 if analysis["tone_summary"] == "Optimistic" else 0.0,
            f"{self.feature_prefix}tone_cautious": 1.0 if analysis["tone_summary"] == "Cautious" else 0.0,
            f"{self.feature_prefix}tone_defensive": 1.0 if analysis["tone_summary"] == "Defensive" else 0.0,
            f"{self.feature_prefix}tone_pessimistic": 1.0 if analysis["tone_summary"] == "Pessimistic" else 0.0,
        }

        return features

    def extract_features_ai(self, transcript: str, symbol: str = "") -> Dict[str, any]:
        """
        Extract features using AI (GPT-4o-mini) for better accuracy.

        This method uses OpenAI to analyze the entire transcript intelligently,
        without relying on speaker name parsing.

        Args:
            transcript: Full earnings call transcript
            symbol: Stock ticker (for context)

        Returns:
            Dict with scores and AI summary
        """
        try:
            client = OpenAI(api_key=settings.openai_api_key)

            # Truncate transcript if too long (GPT-4o-mini limit: ~128k tokens)
            max_chars = 50000  # ~12,500 tokens
            if len(transcript) > max_chars:
                # Take first 60% + last 40% (usually CEO speaks at start and end)
                split_point = int(max_chars * 0.6)
                transcript = transcript[:split_point] + "\n...[middle section omitted]...\n" + transcript[-int(max_chars * 0.4):]

            prompt = f"""Analyze this earnings call transcript for {symbol}.

Focus on CEO/CFO tone, confidence, and forward guidance.

Your task:
1. **Confidence Score** (0-1): How confident is management about the business?
   - 0.8-1.0: Very confident, strong language, no hedging
   - 0.6-0.8: Confident, mostly positive
   - 0.4-0.6: Neutral, balanced
   - 0.2-0.4: Cautious, some concerns
   - 0.0-0.2: Defensive, weak, many hedges

2. **Sentiment Score** (0-1): Overall positive/negative tone
   - 0.8-1.0: Very positive (beat, growth, momentum)
   - 0.6-0.8: Positive
   - 0.4-0.6: Neutral/Mixed
   - 0.2-0.4: Negative (miss, challenges, headwinds)
   - 0.0-0.2: Very negative

3. **Guidance Sentiment** (0-1): Forward-looking statements
   - 0.8-1.0: Raised guidance, very optimistic outlook
   - 0.6-0.8: Positive outlook
   - 0.4-0.6: Maintained guidance, stable
   - 0.2-0.4: Lowered expectations, cautious
   - 0.0-0.2: Downgraded guidance, pessimistic

4. **Uncertainty Ratio** (0-1): Use of hedging/uncertain language
   - 0.0-0.2: Very certain, clear statements
   - 0.2-0.4: Mostly certain
   - 0.4-0.6: Moderate uncertainty
   - 0.6-0.8: High uncertainty
   - 0.8-1.0: Very uncertain, many hedges

5. **Tone Summary**: One word from: Confident, Optimistic, Neutral, Cautious, Defensive, Pessimistic

6. **Executive Summary**: 2-3 sentences summarizing the CEO/CFO's main message and tone.

Transcript:
{transcript}

Respond ONLY with valid JSON (no markdown):
{{
  "confidence_score": 0.75,
  "sentiment_score": 0.82,
  "guidance_sentiment": 0.68,
  "uncertainty_ratio": 0.25,
  "tone_summary": "Confident",
  "executive_summary": "Management expressed strong confidence in AI growth drivers..."
}}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            ai_response = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if ai_response.startswith("```"):
                ai_response = ai_response.split("```")[1]
                if ai_response.startswith("json"):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()

            import json
            result = json.loads(ai_response)

            logger.info(f"AI transcript analysis successful for {symbol}")

            return {
                "has_transcript": True,
                "confidence_score": result.get("confidence_score", 0.5),
                "sentiment_score": result.get("sentiment_score", 0.5),
                "guidance_sentiment": result.get("guidance_sentiment", 0.5),
                "uncertainty_ratio": result.get("uncertainty_ratio", 0.5),
                "tone_summary": result.get("tone_summary", "Neutral"),
                "executive_summary": result.get("executive_summary", "Analysis unavailable"),
                "analysis_method": "AI (GPT-4o-mini)"
            }

        except Exception as e:
            logger.error(f"AI transcript analysis failed: {e}, falling back to keyword analysis")
            # Fallback to original method
            analysis = self.calculate_ceo_sentiment_score(transcript)
            return {
                "has_transcript": True,
                "confidence_score": analysis["ceo_confidence_score"],
                "sentiment_score": analysis["ceo_overall_sentiment"],
                "guidance_sentiment": analysis["forward_guidance_sentiment"],
                "uncertainty_ratio": analysis["uncertainty_ratio"],
                "tone_summary": analysis.get("tone_summary", "Neutral"),
                "executive_summary": "Unable to generate AI summary. Keyword-based analysis used.",
                "analysis_method": "Fallback (Keywords)"
            }
