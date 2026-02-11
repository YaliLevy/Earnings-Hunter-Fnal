"""
CrewAI Crew definition for The Earnings Hunter.

Orchestrates multi-agent analysis using the Golden Triangle framework.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import os

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from src.agents.tools.fmp_tools import ALL_FMP_TOOLS
from src.utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# CRITICAL: Export OPENAI_API_KEY to environment for CrewAI
# CrewAI needs it as an environment variable, not just in Pydantic settings
os.environ["OPENAI_API_KEY"] = settings.openai_api_key


class EarningsHunterCrew:
    """
    CrewAI Crew for earnings analysis.

    Implements the Golden Triangle framework with three specialized agents:
    - Scout Agent: Financial data + CEO tone (40% + 35% = 75%)
    - Social Listener Agent: Reddit sentiment (25%)
    - Fusion Agent: Synthesizes all data into final prediction
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the crew.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

        # Load agent and task configs
        config_dir = Path(__file__).parent / "config"
        self.agents_config = self._load_yaml(config_dir / "agents.yaml")
        self.tasks_config = self._load_yaml(config_dir / "tasks.yaml")

        # Initialize LLM (using gpt-4o-mini for cost efficiency)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=settings.openai_api_key
        )

        # Create agents
        self.scout_agent = self._create_scout_agent()
        self.social_listener_agent = self._create_social_listener_agent()
        self.fusion_agent = self._create_fusion_agent()

        logger.info("EarningsHunterCrew initialized")

    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
            return {}

    def _create_scout_agent(self) -> Agent:
        """Create the Scout Agent for financial and transcript analysis."""
        config = self.agents_config.get("scout_agent", {})

        return Agent(
            role=config.get("role", "Financial Data Scout & CEO Tone Analyst"),
            goal=config.get("goal", "Analyze financial data and CEO tone from transcripts"),
            backstory=config.get("backstory", "Expert financial analyst"),
            verbose=self.verbose,
            allow_delegation=False,
            tools=ALL_FMP_TOOLS,
            llm=self.llm
        )

    def _create_social_listener_agent(self) -> Agent:
        """Create the Social Listener Agent for Reddit analysis."""
        config = self.agents_config.get("social_listener_agent", {})

        return Agent(
            role=config.get("role", "Street Sentiment Analyst"),
            goal=config.get("goal", "Gauge retail investor sentiment from social media"),
            backstory=config.get("backstory", "Expert in behavioral finance"),
            verbose=self.verbose,
            allow_delegation=False,
            tools=ALL_FMP_TOOLS,  # Using FMP social sentiment tools
            llm=self.llm
        )

    def _create_fusion_agent(self) -> Agent:
        """Create the Fusion Agent for synthesizing analysis."""
        config = self.agents_config.get("fusion_agent", {})

        return Agent(
            role=config.get("role", "Chief Investment Analyst"),
            goal=config.get("goal", "Synthesize analysis into final prediction"),
            backstory=config.get("backstory", "Senior investment analyst"),
            verbose=self.verbose,
            allow_delegation=False,
            tools=[],  # Fusion agent doesn't need external tools
            llm=self.llm
        )

    def _create_financial_task(self, symbol: str) -> Task:
        """Create financial analysis task."""
        config = self.tasks_config.get("analyze_financials", {})

        return Task(
            description=config.get("description", "").format(symbol=symbol),
            expected_output=config.get("expected_output", "Financial analysis"),
            agent=self.scout_agent
        )

    def _create_ceo_tone_task(
        self,
        symbol: str,
        year: int,
        quarter: int,
        financial_task: Task
    ) -> Task:
        """Create CEO tone analysis task."""
        config = self.tasks_config.get("analyze_ceo_tone", {})

        return Task(
            description=config.get("description", "").format(
                symbol=symbol,
                year=year,
                quarter=quarter
            ),
            expected_output=config.get("expected_output", "CEO tone analysis"),
            agent=self.scout_agent,
            context=[financial_task]
        )

    def _create_social_task(self, symbol: str) -> Task:
        """Create social sentiment analysis task."""
        config = self.tasks_config.get("analyze_social_sentiment", {})

        return Task(
            description=config.get("description", "").format(symbol=symbol),
            expected_output=config.get("expected_output", "Social sentiment analysis"),
            agent=self.social_listener_agent
        )

    def _create_synthesis_task(
        self,
        symbol: str,
        ml_prediction: str,
        ml_confidence: float,
        financial_task: Task,
        ceo_tone_task: Task,
        social_task: Task
    ) -> Task:
        """Create synthesis task."""
        config = self.tasks_config.get("synthesize_analysis", {})

        return Task(
            description=config.get("description", "").format(
                symbol=symbol,
                ml_prediction=ml_prediction,
                ml_confidence=ml_confidence
            ),
            expected_output=config.get("expected_output", "Complete synthesis"),
            agent=self.fusion_agent,
            context=[financial_task, ceo_tone_task, social_task]
        )

    def analyze(
        self,
        symbol: str,
        year: int,
        quarter: int,
        ml_prediction: str = "Unknown",
        ml_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run full earnings analysis for a symbol.

        Args:
            symbol: Stock ticker symbol
            year: Earnings year
            quarter: Earnings quarter (1-4)
            ml_prediction: ML model prediction (if available)
            ml_confidence: ML model confidence (if available)

        Returns:
            Analysis result dictionary
        """
        logger.info(f"Starting analysis for {symbol} Q{quarter} {year}")

        try:
            # Create tasks
            financial_task = self._create_financial_task(symbol)
            ceo_tone_task = self._create_ceo_tone_task(
                symbol, year, quarter, financial_task
            )
            social_task = self._create_social_task(symbol)
            synthesis_task = self._create_synthesis_task(
                symbol,
                ml_prediction,
                ml_confidence,
                financial_task,
                ceo_tone_task,
                social_task
            )

            # Create and run crew
            crew = Crew(
                agents=[
                    self.scout_agent,
                    self.social_listener_agent,
                    self.fusion_agent
                ],
                tasks=[
                    financial_task,
                    ceo_tone_task,
                    social_task,
                    synthesis_task
                ],
                process=Process.sequential,
                verbose=self.verbose
            )

            # Execute
            result = crew.kickoff()

            logger.info(f"Analysis complete for {symbol}")

            return {
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "status": "success",
                "raw_output": str(result),
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence
            }

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                "symbol": symbol,
                "year": year,
                "quarter": quarter,
                "status": "error",
                "error": str(e)
            }

    def analyze_quick(self, symbol: str) -> Dict[str, Any]:
        """
        Run quick analysis without CEO tone (for testing).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Quick analysis result
        """
        logger.info(f"Starting quick analysis for {symbol}")

        try:
            # Create simplified tasks
            financial_task = self._create_financial_task(symbol)
            social_task = self._create_social_task(symbol)

            # Simplified synthesis without CEO tone
            synthesis_task = Task(
                description=f"""
                Provide a quick analysis synthesis for {symbol} based on:
                1. Financial data analysis
                2. Social sentiment analysis

                Generate a brief prediction (Growth/Stagnation/Risk) with reasoning.
                Include disclaimer that this is for educational purposes only.
                """,
                expected_output="Quick prediction with brief reasoning",
                agent=self.fusion_agent,
                context=[financial_task, social_task]
            )

            crew = Crew(
                agents=[self.scout_agent, self.social_listener_agent, self.fusion_agent],
                tasks=[financial_task, social_task, synthesis_task],
                process=Process.sequential,
                verbose=self.verbose
            )

            result = crew.kickoff()

            return {
                "symbol": symbol,
                "status": "success",
                "analysis_type": "quick",
                "raw_output": str(result)
            }

        except Exception as e:
            logger.error(f"Error in quick analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            }
