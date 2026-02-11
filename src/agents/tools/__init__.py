"""CrewAI tools for The Earnings Hunter."""

from src.agents.tools.fmp_tools import (
    ALL_FMP_TOOLS,
    # Tool classes (CrewAI 1.x BaseTool)
    FetchStockQuoteTool,
    FetchEarningsSurprisesTool,
    FetchIncomeStatementTool,
    FetchEarningsTranscriptTool,
    FetchAnalystDataTool,
    FetchInsiderActivityTool,
    FetchStockNewsTool,
    FetchHistoricalPricesTool,
    FetchLatestEarningsInfoTool,
    FetchSocialSentimentTool,
)

__all__ = [
    "ALL_FMP_TOOLS",
    "FetchStockQuoteTool",
    "FetchEarningsSurprisesTool",
    "FetchIncomeStatementTool",
    "FetchEarningsTranscriptTool",
    "FetchAnalystDataTool",
    "FetchInsiderActivityTool",
    "FetchStockNewsTool",
    "FetchHistoricalPricesTool",
    "FetchLatestEarningsInfoTool",
    "FetchSocialSentimentTool",
]
