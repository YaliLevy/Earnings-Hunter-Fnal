"""
Sentiment Gauge Component.

Displays a circular gauge for market or stock sentiment,
similar to the StockFlow design with bullish/bearish indicators.
"""

from typing import Literal, Optional
import streamlit as st
import plotly.graph_objects as go

from app.config.theme import COLORS, SENTIMENT_CSS, get_plotly_layout


def create_sentiment_gauge(
    value: float,
    title: str = "",
    height: int = 200,
    show_labels: bool = True
) -> go.Figure:
    """
    Create a circular sentiment gauge chart.

    Args:
        value: Sentiment value (0-100, where 0=Bearish, 100=Bullish)
        title: Optional title above gauge
        height: Chart height in pixels
        show_labels: Whether to show Bearish/Neutral/Bullish labels

    Returns:
        Plotly figure
    """
    # Determine color based on value
    if value >= 60:
        color = COLORS['accent_teal']
        label = "BULLISH"
    elif value <= 40:
        color = COLORS['accent_red']
        label = "BEARISH"
    else:
        color = COLORS['accent_yellow']
        label = "NEUTRAL"

    fig = go.Figure()

    # Main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            'suffix': '%',
            'font': {'size': 36, 'color': color, 'family': 'Inter, sans-serif'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': COLORS['border'],
                'tickfont': {'color': COLORS['text_secondary'], 'size': 10},
                'dtick': 25,
            },
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': COLORS['bg_dark'],
            'borderwidth': 2,
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [0, 40], 'color': f"rgba({int(COLORS['accent_red'][1:3], 16)}, {int(COLORS['accent_red'][3:5], 16)}, {int(COLORS['accent_red'][5:7], 16)}, 0.15)"},
                {'range': [40, 60], 'color': f"rgba({int(COLORS['accent_yellow'][1:3], 16)}, {int(COLORS['accent_yellow'][3:5], 16)}, {int(COLORS['accent_yellow'][5:7], 16)}, 0.15)"},
                {'range': [60, 100], 'color': f"rgba({int(COLORS['accent_teal'][1:3], 16)}, {int(COLORS['accent_teal'][3:5], 16)}, {int(COLORS['accent_teal'][5:7], 16)}, 0.15)"},
            ],
            'threshold': {
                'line': {'color': COLORS['text_primary'], 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        },
        domain={'x': [0, 1], 'y': [0.15, 1]}
    ))

    # Update layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_primary'], 'family': 'Inter, sans-serif'},
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=40),
        title={
            'text': title,
            'font': {'size': 14, 'color': COLORS['text_secondary']},
            'x': 0.5,
            'xanchor': 'center'
        } if title else None
    )

    # Add label annotation
    fig.add_annotation(
        text=label,
        x=0.5,
        y=0.05,
        showarrow=False,
        font=dict(size=14, color=color, family='Inter, sans-serif'),
        xref='paper',
        yref='paper'
    )

    return fig


def display_sentiment_panel(
    value: float,
    title: str = "MARKET SENTIMENT",
    description: Optional[str] = None
):
    """
    Display a complete sentiment panel with gauge and description.

    Args:
        value: Sentiment value (0-100)
        title: Panel title
        description: Optional description text below gauge
    """
    st.markdown(SENTIMENT_CSS, unsafe_allow_html=True)

    # Determine sentiment class
    if value >= 60:
        sentiment_class = "sentiment-bullish"
        sentiment_label = "BULLISH"
    elif value <= 40:
        sentiment_class = "sentiment-bearish"
        sentiment_label = "BEARISH"
    else:
        sentiment_class = "sentiment-neutral"
        sentiment_label = "NEUTRAL"

    # Create gauge
    fig = create_sentiment_gauge(value, height=180)

    # Panel container
    st.markdown(f"""
    <div class="sentiment-panel">
        <div class="sentiment-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)

    # Display gauge
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Add labels below gauge
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: space-between;
        padding: 0 20px;
        margin-top: -20px;
    ">
        <span style="color: {COLORS['accent_red']}; font-size: 11px; font-weight: 500;">BEARISH</span>
        <span style="color: {COLORS['accent_yellow']}; font-size: 11px; font-weight: 500;">NEUTRAL</span>
        <span style="color: {COLORS['accent_teal']}; font-size: 11px; font-weight: 500;">BULLISH</span>
    </div>
    """, unsafe_allow_html=True)

    # Description
    if description:
        st.markdown(f"""
        <div class="sentiment-description" style="text-align: center; padding: 0 16px;">
            {description}
        </div>
        """, unsafe_allow_html=True)


def display_mini_sentiment(value: float, label: str = "Sentiment"):
    """
    Display a compact sentiment indicator.

    Args:
        value: Sentiment value (0-100)
        label: Label text
    """
    # Determine color
    if value >= 60:
        color = COLORS['accent_teal']
        text = "Bullish"
    elif value <= 40:
        color = COLORS['accent_red']
        text = "Bearish"
    else:
        color = COLORS['accent_yellow']
        text = "Neutral"

    html = f"""
    <div style="
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <span style="color: {COLORS['text_secondary']}; font-size: 13px;">{label}</span>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: {color}; font-size: 18px; font-weight: 700;">{value:.0f}%</span>
            <span style="color: {color}; font-size: 12px; font-weight: 500;">{text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_dual_gauge(
    bullish: float,
    bearish: float,
    title: str = "Market Sentiment",
    height: int = 150
) -> go.Figure:
    """
    Create a dual progress bar showing bullish vs bearish sentiment.

    Args:
        bullish: Bullish percentage (0-100)
        bearish: Bearish percentage (0-100)
        title: Chart title
        height: Chart height

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Bullish bar
    fig.add_trace(go.Bar(
        y=[title],
        x=[bullish],
        orientation='h',
        marker=dict(color=COLORS['accent_teal']),
        name='Bullish',
        text=[f'{bullish:.0f}%'],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    # Bearish bar
    fig.add_trace(go.Bar(
        y=[title],
        x=[-bearish],
        orientation='h',
        marker=dict(color=COLORS['accent_red']),
        name='Bearish',
        text=[f'{bearish:.0f}%'],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.update_layout(
        barmode='relative',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text_secondary'])
        ),
        xaxis=dict(
            range=[-100, 100],
            showgrid=False,
            zeroline=True,
            zerolinecolor=COLORS['border'],
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False
        )
    )

    return fig


def display_golden_triangle_sentiment(
    financial_score: float,
    ceo_score: float,
    social_score: float
):
    """
    Display sentiment breakdown for Golden Triangle components.

    Args:
        financial_score: Financial data score (0-10)
        ceo_score: CEO tone score (0-10)
        social_score: Social sentiment score (0-10)
    """
    st.markdown(SENTIMENT_CSS, unsafe_allow_html=True)

    # Convert to percentages (0-100)
    fin_pct = financial_score * 10
    ceo_pct = ceo_score * 10
    social_pct = social_score * 10

    # Weighted average
    weighted = (fin_pct * 0.4) + (ceo_pct * 0.35) + (social_pct * 0.25)

    scores = [
        ("Financial (40%)", fin_pct, COLORS['accent_blue']),
        ("CEO Tone (35%)", ceo_pct, COLORS['accent_purple']),
        ("Social (25%)", social_pct, COLORS['accent_teal']),
    ]

    html = f"""
    <div class="sentiment-panel">
        <div class="sentiment-title">GOLDEN TRIANGLE SCORES</div>
    """

    for label, score, color in scores:
        bar_width = min(score, 100)
        html += f"""
        <div style="margin: 12px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: {COLORS['text_secondary']}; font-size: 12px;">{label}</span>
                <span style="color: {color}; font-size: 14px; font-weight: 600;">{score:.0f}%</span>
            </div>
            <div style="
                background: {COLORS['bg_dark']};
                border-radius: 4px;
                height: 8px;
                overflow: hidden;
            ">
                <div style="
                    background: {color};
                    width: {bar_width}%;
                    height: 100%;
                    border-radius: 4px;
                    transition: width 0.5s ease;
                "></div>
            </div>
        </div>
        """

    # Weighted total
    total_color = COLORS['accent_teal'] if weighted >= 60 else (COLORS['accent_red'] if weighted < 40 else COLORS['accent_yellow'])
    html += f"""
        <div style="
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid {COLORS['border']};
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span style="color: {COLORS['text_primary']}; font-size: 14px; font-weight: 600;">Weighted Total</span>
            <span style="color: {total_color}; font-size: 24px; font-weight: 700;">{weighted:.0f}%</span>
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)
