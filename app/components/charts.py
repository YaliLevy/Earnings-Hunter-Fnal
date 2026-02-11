"""
Chart components for The Earnings Hunter dashboard.

Provides Plotly visualizations for Golden Triangle analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import pandas as pd


def create_golden_triangle_radar(
    financial_score: float,
    ceo_tone_score: float,
    social_score: float,
    title: str = "Golden Triangle Analysis"
) -> go.Figure:
    """
    Create radar chart for Golden Triangle scores.

    Args:
        financial_score: Financial metrics score (0-10)
        ceo_tone_score: CEO tone analysis score (0-10)
        social_score: Social sentiment score (0-10)
        title: Chart title

    Returns:
        Plotly figure object
    """
    categories = ["📊 Financial (40%)", "🎤 CEO Tone (35%)", "📱 Social (25%)"]

    fig = go.Figure()

    # Add the scores trace
    fig.add_trace(go.Scatterpolar(
        r=[financial_score, ceo_tone_score, social_score],
        theta=categories,
        fill='toself',
        fillcolor='rgba(78, 205, 196, 0.3)',
        line=dict(color='#4ECDC4', width=2),
        name='Scores'
    ))

    # Add max reference
    fig.add_trace(go.Scatterpolar(
        r=[10, 10, 10],
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 255, 255, 0.05)',
        line=dict(color='#333', width=1, dash='dot'),
        name='Max'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(color='#888'),
                gridcolor='#333'
            ),
            angularaxis=dict(
                tickfont=dict(color='#fff', size=12),
                gridcolor='#333'
            ),
            bgcolor='#1a1a2e'
        ),
        showlegend=False,
        title=dict(
            text=title,
            font=dict(color='#fff', size=16)
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        height=400
    )

    return fig


def create_confidence_gauge(
    confidence: float,
    prediction: str,
    title: str = "Prediction Confidence"
) -> go.Figure:
    """
    Create gauge chart for prediction confidence.

    Args:
        confidence: Confidence value (0-1)
        prediction: Prediction label
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Color based on prediction
    color_map = {
        "Growth": "#4ECDC4",
        "Stagnation": "#FFE66D",
        "Risk": "#FF6B6B"
    }
    color = color_map.get(prediction, "#4ECDC4")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}},
        title={'text': f"{prediction}", 'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#888',
                'tickfont': {'color': '#888'}
            },
            'bar': {'color': color},
            'bgcolor': '#333',
            'borderwidth': 2,
            'bordercolor': '#444',
            'steps': [
                {'range': [0, 50], 'color': '#2a2a3e'},
                {'range': [50, 70], 'color': '#3a3a4e'},
                {'range': [70, 100], 'color': '#4a4a5e'}
            ],
            'threshold': {
                'line': {'color': '#fff', 'width': 2},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font={'color': '#fff'},
        height=300
    )

    return fig


def create_model_comparison_bar(
    model_predictions: Dict[str, str],
    title: str = "Model Predictions"
) -> go.Figure:
    """
    Create horizontal bar chart showing each model's prediction.

    Args:
        model_predictions: Dict mapping model name to prediction
        title: Chart title

    Returns:
        Plotly figure object
    """
    models = list(model_predictions.keys())
    predictions = list(model_predictions.values())

    # Color map
    color_map = {
        "Growth": "#4ECDC4",
        "Stagnation": "#FFE66D",
        "Risk": "#FF6B6B"
    }
    colors = [color_map.get(p, "#888") for p in predictions]

    fig = go.Figure(go.Bar(
        y=models,
        x=[1] * len(models),  # All bars same length
        orientation='h',
        marker=dict(color=colors),
        text=predictions,
        textposition='inside',
        textfont=dict(color='#000', size=14)
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#fff', size=16)
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        xaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(color='#fff', size=12),
            showgrid=False
        ),
        height=250,
        margin=dict(l=150, r=20, t=40, b=20)
    )

    return fig


def create_sentiment_timeline(
    dates: List[str],
    sentiments: List[float],
    title: str = "Sentiment Over Time"
) -> go.Figure:
    """
    Create line chart for sentiment over time.

    Args:
        dates: List of date strings
        sentiments: List of sentiment values (-1 to 1)
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=dates,
        y=sentiments,
        mode='lines+markers',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(size=8, color='#4ECDC4'),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.2)',
        name='Sentiment'
    ))

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#888",
        annotation_text="Neutral",
        annotation_position="right"
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#fff', size=16)
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        xaxis=dict(
            tickfont=dict(color='#888'),
            gridcolor='#333',
            title='Date'
        ),
        yaxis=dict(
            tickfont=dict(color='#888'),
            gridcolor='#333',
            title='Sentiment',
            range=[-1, 1]
        ),
        height=300,
        showlegend=False
    )

    return fig


def create_earnings_surprise_bar(
    eps_actual: float,
    eps_estimate: float,
    revenue_actual: float,
    revenue_estimate: float,
    title: str = "Earnings Surprises"
) -> go.Figure:
    """
    Create grouped bar chart for earnings surprises.

    Args:
        eps_actual: Actual EPS
        eps_estimate: Estimated EPS
        revenue_actual: Actual revenue (billions)
        revenue_estimate: Estimated revenue (billions)
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('EPS', 'Revenue ($B)'),
        horizontal_spacing=0.15
    )

    # EPS bars
    eps_color = '#4ECDC4' if eps_actual >= eps_estimate else '#FF6B6B'
    fig.add_trace(
        go.Bar(
            x=['Estimate', 'Actual'],
            y=[eps_estimate, eps_actual],
            marker_color=['#888', eps_color],
            text=[f'${eps_estimate:.2f}', f'${eps_actual:.2f}'],
            textposition='outside',
            textfont=dict(color='#fff')
        ),
        row=1, col=1
    )

    # Revenue bars
    rev_color = '#4ECDC4' if revenue_actual >= revenue_estimate else '#FF6B6B'
    fig.add_trace(
        go.Bar(
            x=['Estimate', 'Actual'],
            y=[revenue_estimate, revenue_actual],
            marker_color=['#888', rev_color],
            text=[f'${revenue_estimate:.1f}B', f'${revenue_actual:.1f}B'],
            textposition='outside',
            textfont=dict(color='#fff')
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#fff', size=16)
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        showlegend=False,
        height=300
    )

    fig.update_xaxes(tickfont=dict(color='#888'), gridcolor='#333')
    fig.update_yaxes(tickfont=dict(color='#888'), gridcolor='#333')

    # Update subplot title colors
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#fff', size=14)

    return fig


def create_hype_meter(
    hype_index: float,
    title: str = "Social Hype Index"
) -> go.Figure:
    """
    Create bullet/progress chart for hype index.

    Args:
        hype_index: Hype value (0-100)
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Determine color based on hype level
    if hype_index >= 70:
        color = "#FF6B6B"  # High hype (potentially risky)
        level = "🔥 High"
    elif hype_index >= 40:
        color = "#FFE66D"  # Moderate
        level = "📊 Moderate"
    else:
        color = "#4ECDC4"  # Low
        level = "😴 Low"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hype_index,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '/100', 'font': {'size': 36, 'color': '#fff'}},
        title={'text': f"Hype Level: {level}", 'font': {'size': 16, 'color': color}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#888',
                'tickfont': {'color': '#888'}
            },
            'bar': {'color': color},
            'bgcolor': '#333',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(78, 205, 196, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(255, 230, 109, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 107, 107, 0.2)'}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        height=250
    )

    return fig


def create_accuracy_pie(
    correct: int,
    incorrect: int,
    pending: int,
    title: str = "Prediction Accuracy"
) -> go.Figure:
    """
    Create pie chart for prediction accuracy.

    Args:
        correct: Number of correct predictions
        incorrect: Number of incorrect predictions
        pending: Number of pending predictions
        title: Chart title

    Returns:
        Plotly figure object
    """
    labels = ['Correct', 'Incorrect', 'Pending']
    values = [correct, incorrect, pending]
    colors = ['#4ECDC4', '#FF6B6B', '#FFE66D']

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='percent+label',
        textfont=dict(color='#fff', size=12)
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#fff', size=16)
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        showlegend=True,
        legend=dict(
            font=dict(color='#fff'),
            orientation='h',
            yanchor='bottom',
            y=-0.2
        ),
        height=350
    )

    return fig


def create_feature_importance_bar(
    features: List[tuple],
    title: str = "Top Feature Importance"
) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.

    Args:
        features: List of (feature_name, importance) tuples
        title: Chart title

    Returns:
        Plotly figure object
    """
    feature_names = [f[0] for f in features]
    importances = [f[1] for f in features]

    # Color gradient based on importance
    colors = px.colors.sequential.Teal[:len(features)]

    fig = go.Figure(go.Bar(
        y=feature_names,
        x=importances,
        orientation='h',
        marker=dict(
            color=importances,
            colorscale='Teal'
        ),
        text=[f'{v:.1%}' for v in importances],
        textposition='outside',
        textfont=dict(color='#fff')
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#fff', size=16)
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        xaxis=dict(
            tickfont=dict(color='#888'),
            gridcolor='#333',
            title='Importance',
            tickformat='.0%'
        ),
        yaxis=dict(
            tickfont=dict(color='#fff', size=11),
            showgrid=False,
            categoryorder='total ascending'
        ),
        height=max(250, len(features) * 30),
        margin=dict(l=200, r=50, t=40, b=40)
    )

    return fig
