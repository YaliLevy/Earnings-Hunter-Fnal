"""Agent module for The Earnings Hunter - CrewAI multi-agent system."""

from src.agents.orchestrator import EarningsOrchestrator
from src.agents.insight_generator import InsightGenerator

# CrewAI crew is imported lazily in analysis.py when deep analysis is requested
# This avoids crewai.tools.BaseTool import errors at startup

__all__ = [
    "EarningsOrchestrator",
    "InsightGenerator",
]
