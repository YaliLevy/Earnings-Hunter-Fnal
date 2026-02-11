"""
Market Status Component.

Displays whether the US stock market (NYSE/NASDAQ) is currently open or closed,
with countdown timer to next open/close.
"""

from datetime import datetime, time, timedelta
import pytz
import streamlit as st

from app.config.theme import COLORS, MARKET_STATUS_CSS


# US Eastern timezone (NYSE/NASDAQ)
ET = pytz.timezone('US/Eastern')

# Market hours (Eastern Time)
MARKET_OPEN = time(9, 30)   # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET

# US Market holidays 2026 (NYSE/NASDAQ)
MARKET_HOLIDAYS_2026 = [
    datetime(2026, 1, 1),    # New Year's Day
    datetime(2026, 1, 19),   # MLK Day
    datetime(2026, 2, 16),   # Presidents' Day
    datetime(2026, 4, 3),    # Good Friday
    datetime(2026, 5, 25),   # Memorial Day
    datetime(2026, 7, 3),    # Independence Day (observed)
    datetime(2026, 9, 7),    # Labor Day
    datetime(2026, 11, 26),  # Thanksgiving
    datetime(2026, 12, 25),  # Christmas
]


def get_current_et_time() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def is_market_holiday(date: datetime) -> bool:
    """Check if given date is a market holiday."""
    date_only = date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    return date_only in MARKET_HOLIDAYS_2026


def is_weekend(date: datetime) -> bool:
    """Check if given date is a weekend."""
    return date.weekday() >= 5  # 5 = Saturday, 6 = Sunday


def is_market_open() -> bool:
    """
    Check if the US stock market is currently open.

    Market hours: 9:30 AM - 4:00 PM Eastern Time
    Closed on weekends and market holidays.

    Returns:
        True if market is open, False otherwise
    """
    now = get_current_et_time()

    # Check if weekend
    if is_weekend(now):
        return False

    # Check if holiday
    if is_market_holiday(now):
        return False

    # Check if within market hours
    current_time = now.time()
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def get_next_market_event() -> tuple:
    """
    Get the next market open or close event.

    Returns:
        Tuple of (event_type, event_datetime)
        event_type: "open" or "close"
    """
    now = get_current_et_time()
    current_time = now.time()

    # If market is currently open, next event is close
    if is_market_open():
        close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return ("close", close_time)

    # Market is closed - find next open
    # Start with today at market open
    next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

    # If we're past market open today, move to tomorrow
    if current_time >= MARKET_OPEN:
        next_open += timedelta(days=1)

    # Skip weekends and holidays
    while is_weekend(next_open) or is_market_holiday(next_open):
        next_open += timedelta(days=1)

    return ("open", next_open)


def get_time_until_event() -> str:
    """
    Get human-readable time until next market event.

    Returns:
        String like "2h 30m" or "Opens Monday 9:30 AM"
    """
    now = get_current_et_time()
    event_type, event_time = get_next_market_event()

    delta = event_time - now
    total_seconds = delta.total_seconds()

    if total_seconds < 0:
        return "Now"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    if hours >= 24:
        # More than a day - show day name
        day_name = event_time.strftime("%A")
        time_str = event_time.strftime("%-I:%M %p")
        verb = "Opens" if event_type == "open" else "Closes"
        return f"{verb} {day_name} {time_str}"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def display_market_status():
    """
    Display market status badge in Streamlit.

    Shows:
    - Green "Market Open" badge when market is open
    - Red "Market Closed" badge when market is closed
    - Countdown timer to next event
    """
    # Inject CSS
    st.markdown(MARKET_STATUS_CSS, unsafe_allow_html=True)

    market_open = is_market_open()
    time_str = get_time_until_event()

    if market_open:
        status_class = "market-open"
        dot_class = "status-dot-open"
        status_text = "Market Open"
        sub_text = f"Closes in {time_str}"
    else:
        status_class = "market-closed"
        dot_class = "status-dot-closed"
        status_text = "Market Closed"
        sub_text = time_str

    html = f"""
    <div style="display: flex; align-items: center; gap: 8px;">
        <div class="market-status {status_class}">
            <span class="status-dot {dot_class}"></span>
            <span>{status_text}</span>
        </div>
        <span style="color: {COLORS['text_secondary']}; font-size: 12px;">
            {sub_text}
        </span>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def display_market_status_compact():
    """
    Display compact market status (just the badge, no timer).
    """
    st.markdown(MARKET_STATUS_CSS, unsafe_allow_html=True)

    market_open = is_market_open()

    if market_open:
        html = f"""
        <div class="market-status market-open">
            <span class="status-dot status-dot-open"></span>
            <span>Market Open</span>
        </div>
        """
    else:
        html = f"""
        <div class="market-status market-closed">
            <span class="status-dot status-dot-closed"></span>
            <span>Market Closed</span>
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)


def get_market_session_info() -> dict:
    """
    Get detailed market session information.

    Returns:
        Dict with market session details
    """
    now = get_current_et_time()
    is_open = is_market_open()
    event_type, event_time = get_next_market_event()

    return {
        "is_open": is_open,
        "current_time_et": now.strftime("%I:%M %p ET"),
        "current_date": now.strftime("%B %d, %Y"),
        "is_weekend": is_weekend(now),
        "is_holiday": is_market_holiday(now),
        "next_event": event_type,
        "next_event_time": event_time.strftime("%I:%M %p ET"),
        "time_until_event": get_time_until_event(),
        "market_hours": f"{MARKET_OPEN.strftime('%I:%M %p')} - {MARKET_CLOSE.strftime('%I:%M %p')} ET"
    }
