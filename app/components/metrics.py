"""
Metric card components for The Earnings Hunter dashboard.

Provides reusable metric display components.
"""

import streamlit as st
from typing import Optional, Literal


def metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: Literal["positive", "negative", "neutral"] = "neutral"
) -> None:
    """
    Display a styled metric card.

    Args:
        title: Card title
        value: Main value to display
        subtitle: Optional subtitle text
        delta: Optional delta/change value
        delta_color: Color for delta (positive=green, negative=red, neutral=gray)
    """
    delta_class = {
        "positive": "delta-positive",
        "negative": "delta-negative",
        "neutral": ""
    }.get(delta_color, "")

    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    subtitle_html = f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {subtitle_html}
        {delta_html}
    </div>
    <style>
        .metric-card {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #333;
        }}
        .metric-title {{
            color: #4ECDC4;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .metric-value {{
            color: #fff;
            font-size: 32px;
            font-weight: bold;
        }}
        .metric-subtitle {{
            color: #888;
            font-size: 12px;
            margin-top: 5px;
        }}
        .metric-delta {{
            font-size: 14px;
            margin-top: 5px;
        }}
        .delta-positive {{ color: #4ECDC4; }}
        .delta-negative {{ color: #FF6B6B; }}
    </style>
    """, unsafe_allow_html=True)


def score_card(
    label: str,
    score: float,
    max_score: float = 10.0,
    color: Optional[str] = None
) -> None:
    """
    Display a score card with visual indicator.

    Args:
        label: Score label
        score: Score value
        max_score: Maximum possible score
        color: Optional override color
    """
    # Determine color based on score percentage
    pct = score / max_score
    if color is None:
        if pct >= 0.7:
            color = "#4ECDC4"  # Green/teal
        elif pct >= 0.4:
            color = "#FFE66D"  # Yellow
        else:
            color = "#FF6B6B"  # Red

    st.markdown(f"""
    <div class="score-card">
        <div class="score-value" style="color: {color}">{score:.1f}</div>
        <div class="score-label">{label}</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: {pct*100}%; background: {color}"></div>
        </div>
    </div>
    <style>
        .score-card {{
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid #333;
        }}
        .score-value {{
            font-size: 36px;
            font-weight: bold;
        }}
        .score-label {{
            color: #888;
            font-size: 14px;
            margin-top: 5px;
        }}
        .score-bar-bg {{
            background: #333;
            border-radius: 5px;
            height: 6px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .score-bar-fill {{
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease;
        }}
    </style>
    """, unsafe_allow_html=True)


def prediction_badge(
    prediction: str,
    confidence: float
) -> None:
    """
    Display a prediction badge with confidence.

    Args:
        prediction: Prediction label (Growth/Stagnation/Risk)
        confidence: Confidence value (0-1)
    """
    color_map = {
        "Growth": ("#4ECDC4", "📈"),
        "Stagnation": ("#FFE66D", "📊"),
        "Risk": ("#FF6B6B", "📉")
    }

    color, icon = color_map.get(prediction, ("#888", "❓"))

    st.markdown(f"""
    <div class="prediction-badge" style="border-color: {color}">
        <div class="prediction-icon">{icon}</div>
        <div class="prediction-label" style="color: {color}">{prediction}</div>
        <div class="prediction-confidence">{confidence:.0%} confidence</div>
    </div>
    <style>
        .prediction-badge {{
            background: #1a1a2e;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 3px solid;
            display: inline-block;
            min-width: 200px;
        }}
        .prediction-icon {{
            font-size: 48px;
            margin-bottom: 10px;
        }}
        .prediction-label {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .prediction-confidence {{
            color: #888;
            font-size: 16px;
        }}
    </style>
    """, unsafe_allow_html=True)


def insight_box(
    title: str,
    content: str,
    box_type: Literal["insight", "risk", "highlight"] = "insight"
) -> None:
    """
    Display an insight box with styled border.

    Args:
        title: Box title
        content: Box content
        box_type: Type of box (insight=teal, risk=red, highlight=yellow)
    """
    color_map = {
        "insight": "#4ECDC4",
        "risk": "#FF6B6B",
        "highlight": "#FFE66D"
    }
    color = color_map.get(box_type, "#4ECDC4")

    st.markdown(f"""
    <div class="insight-box" style="border-left-color: {color}">
        <div class="insight-title" style="color: {color}">{title}</div>
        <div class="insight-content">{content}</div>
    </div>
    <style>
        .insight-box {{
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border-left: 4px solid;
        }}
        .insight-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .insight-content {{
            color: #ccc;
            font-size: 14px;
            line-height: 1.6;
        }}
    </style>
    """, unsafe_allow_html=True)


def golden_triangle_mini(
    financial: float,
    ceo_tone: float,
    social: float
) -> None:
    """
    Display a compact Golden Triangle score summary.

    Args:
        financial: Financial score (0-10)
        ceo_tone: CEO tone score (0-10)
        social: Social score (0-10)
    """
    weighted = (financial * 0.4) + (ceo_tone * 0.35) + (social * 0.25)

    st.markdown(f"""
    <div class="triangle-mini">
        <div class="triangle-item">
            <span class="triangle-emoji">📊</span>
            <span class="triangle-score">{financial:.1f}</span>
            <span class="triangle-weight">40%</span>
        </div>
        <div class="triangle-item">
            <span class="triangle-emoji">🎤</span>
            <span class="triangle-score">{ceo_tone:.1f}</span>
            <span class="triangle-weight">35%</span>
        </div>
        <div class="triangle-item">
            <span class="triangle-emoji">📱</span>
            <span class="triangle-score">{social:.1f}</span>
            <span class="triangle-weight">25%</span>
        </div>
        <div class="triangle-total">
            <span class="triangle-emoji">🎯</span>
            <span class="triangle-score">{weighted:.1f}</span>
            <span class="triangle-weight">Total</span>
        </div>
    </div>
    <style>
        .triangle-mini {{
            display: flex;
            justify-content: space-around;
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #333;
        }}
        .triangle-item {{
            text-align: center;
        }}
        .triangle-total {{
            text-align: center;
            border-left: 1px solid #333;
            padding-left: 20px;
        }}
        .triangle-emoji {{
            display: block;
            font-size: 24px;
            margin-bottom: 5px;
        }}
        .triangle-score {{
            display: block;
            color: #4ECDC4;
            font-size: 20px;
            font-weight: bold;
        }}
        .triangle-weight {{
            display: block;
            color: #666;
            font-size: 12px;
        }}
    </style>
    """, unsafe_allow_html=True)


def model_agreement_display(
    models_agree: int,
    models_total: int,
    model_predictions: dict
) -> None:
    """
    Display model agreement summary.

    Args:
        models_agree: Number of models that agree on prediction
        models_total: Total number of models
        model_predictions: Dict of model_name -> prediction
    """
    agreement_pct = models_agree / models_total

    # Color based on agreement level
    if agreement_pct >= 0.8:
        color = "#4ECDC4"
        label = "Strong Consensus"
    elif agreement_pct >= 0.6:
        color = "#FFE66D"
        label = "Moderate Agreement"
    else:
        color = "#FF6B6B"
        label = "Mixed Signals"

    st.markdown(f"""
    <div class="agreement-box">
        <div class="agreement-header">
            <span class="agreement-count" style="color: {color}">{models_agree}/{models_total}</span>
            <span class="agreement-label">{label}</span>
        </div>
        <div class="model-list">
    """, unsafe_allow_html=True)

    for model, pred in model_predictions.items():
        pred_color = {"Growth": "#4ECDC4", "Stagnation": "#FFE66D", "Risk": "#FF6B6B"}.get(pred, "#888")
        st.markdown(f"""
            <div class="model-item">
                <span class="model-name">{model}</span>
                <span class="model-pred" style="color: {pred_color}">{pred}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        </div>
    </div>
    <style>
        .agreement-box {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #333;
        }
        .agreement-header {
            text-align: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        .agreement-count {
            font-size: 36px;
            font-weight: bold;
            display: block;
        }
        .agreement-label {
            color: #888;
            font-size: 14px;
        }
        .model-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .model-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 10px;
            background: #16213e;
            border-radius: 5px;
        }
        .model-name {
            color: #ccc;
            font-size: 14px;
        }
        .model-pred {
            font-weight: bold;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)
