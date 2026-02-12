"""
Historical data router - Price history for charts.

Provides historical price data for the chart component.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta

from src.data_ingestion.fmp_client import FMPClient
from config.settings import get_settings

router = APIRouter()

# Initialize FMP client
_fmp_client: Optional[FMPClient] = None


def get_fmp_client() -> FMPClient:
    """Get or create FMP client instance."""
    global _fmp_client
    if _fmp_client is None:
        settings = get_settings()
        _fmp_client = FMPClient(settings.fmp_api_key)
    return _fmp_client


class PricePoint(BaseModel):
    """Historical price data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None


class HistoricalResponse(BaseModel):
    """Historical prices response."""
    symbol: str
    timeframe: str
    prices: List[PricePoint]
    sma20: List[Optional[float]]
    forecast: List[PricePoint]


@router.get("/historical/{symbol}", response_model=HistoricalResponse)
async def get_historical_prices(
    symbol: str,
    days: int = Query(90, description="Number of days of history"),
    timeframe: str = Query("3M", description="Timeframe label (1W, 1M, 3M, 6M, 1Y)")
):
    """
    Get historical price data for charts.

    Returns OHLC data with calculated SMA(20) and 7-day linear projection.
    """
    symbol = symbol.upper().strip()

    # Convert timeframe to days if provided
    timeframe_days = {
        "1W": 7,
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365
    }
    days = timeframe_days.get(timeframe, days)

    try:
        fmp_client = get_fmp_client()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        prices = fmp_client.get_historical_prices(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        if not prices:
            raise HTTPException(status_code=404, detail=f"No price history for {symbol}")

        # Convert to response format
        price_points = []
        closes = []

        for p in prices:
            price_points.append({
                "date": str(p.date),
                "open": p.open_price,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
                "change": p.change,
                "change_percent": p.change_percent
            })
            closes.append(p.close)

        # Calculate SMA(20)
        sma20 = _calculate_sma(closes, 20)

        # Calculate 7-day linear projection
        forecast = _calculate_forecast(prices[-1] if prices else None, closes, 7)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "prices": price_points,
            "sma20": sma20,
            "forecast": forecast
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")


def _calculate_sma(closes: List[float], window: int = 20) -> List[Optional[float]]:
    """Calculate Simple Moving Average."""
    sma = []
    for i in range(len(closes)):
        if i < window - 1:
            sma.append(None)
        else:
            avg = sum(closes[i - window + 1:i + 1]) / window
            sma.append(round(avg, 2))
    return sma


def _calculate_forecast(
    last_price,
    closes: List[float],
    days: int = 7
) -> List[dict]:
    """Calculate simple linear projection."""
    if not closes or len(closes) < 5 or not last_price:
        return []

    # Use last 5 days to calculate slope
    recent = closes[-5:]
    slope = (recent[-1] - recent[0]) / 4

    last_date = datetime.strptime(str(last_price.date), "%Y-%m-%d")
    forecast = []

    for i in range(1, days + 1):
        proj_date = last_date + timedelta(days=i)
        proj_price = closes[-1] + (slope * i)

        forecast.append({
            "date": proj_date.strftime("%Y-%m-%d"),
            "open": proj_price,
            "high": proj_price,
            "low": proj_price,
            "close": round(proj_price, 2),
            "volume": None,
            "change": None,
            "change_percent": None
        })

    return forecast
