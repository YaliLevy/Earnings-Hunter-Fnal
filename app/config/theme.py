"""
Earnings Hunter - Premium UI Design System v3.0

Inspired by modern trading platforms with:
- Deep void backgrounds (#050505)
- Signal green accents (#00F090)
- Glass morphism effects
- Smooth animations
"""

# =============================================================================
# COLOR PALETTE - PREMIUM TRADING PLATFORM
# =============================================================================

COLORS = {
    # Backgrounds - Deep Void
    'bg_dark': '#050505',           # Main background (deepest)
    'bg_card': '#0F1115',           # Card background (Obsidian Glass)
    'bg_card_hover': '#1A1D24',     # Card hover state
    'bg_glass': 'rgba(15, 17, 21, 0.85)',  # Glass effect
    'bg_input': '#1F2937',          # Input field background

    # Gradients
    'gradient_primary': 'linear-gradient(135deg, #00F090 0%, #06b6d4 100%)',
    'gradient_success': 'linear-gradient(135deg, #00F090 0%, #10b981 100%)',
    'gradient_danger': 'linear-gradient(135deg, #FF2E50 0%, #f45c43 100%)',
    'gradient_purple': 'linear-gradient(135deg, #8B5CF6 0%, #a78bfa 100%)',
    'gradient_hero': 'linear-gradient(135deg, #050505 0%, #0F1115 50%, #050505 100%)',

    # Accents - Vibrant
    'accent_green': '#00F090',      # Signal Green - Primary positive
    'accent_teal': '#10b981',       # Secondary positive
    'accent_red': '#FF2E50',        # Crimson Laser - Negative
    'accent_yellow': '#f59e0b',     # Warning/neutral
    'accent_purple': '#8B5CF6',     # Deep Reasoning mode
    'accent_cyan': '#06b6d4',       # Secondary accent
    'accent_blue': '#6366f1',       # Primary accent

    # Text
    'text_primary': '#FFFFFF',      # Main text (pure white)
    'text_secondary': '#9ca3af',    # Muted text
    'text_muted': '#6B7280',        # Very muted text
    'text_dark': '#4B5563',         # Darkest text
    'text_tertiary': '#4B5563',     # Alias for text_dark (backward compatibility)

    # Borders
    'border': '#2A2F3A',            # Standard border
    'border_light': '#1F2937',      # Subtle border
    'border_glow_green': 'rgba(0, 240, 144, 0.4)',   # Green glow
    'border_glow_purple': 'rgba(139, 92, 246, 0.4)', # Purple glow

    # Chart colors
    'chart_green': '#00F090',
    'chart_red': '#FF2E50',
    'chart_blue': '#6366f1',
    'chart_grid': '#1F2937',
}

# =============================================================================
# MASTER CSS - PREMIUM TRADING PLATFORM STYLES
# =============================================================================

MASTER_CSS = f"""
<style>
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {{
        --bg-dark: {COLORS['bg_dark']};
        --bg-card: {COLORS['bg_card']};
        --accent-green: {COLORS['accent_green']};
        --accent-red: {COLORS['accent_red']};
        --accent-purple: {COLORS['accent_purple']};
        --text-primary: {COLORS['text_primary']};
        --text-muted: {COLORS['text_muted']};
        --border: {COLORS['border']};
    }}

    /* ===== BASE STYLES ===== */
    .stApp {{
        background: {COLORS['bg_dark']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Hide Streamlit elements */
    #MainMenu, footer, .stDeployButton {{
        display: none !important;
    }}

    [data-testid="stSidebar"] {{
        display: none !important;
    }}

    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}

    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar {{
        width: 4px;
        height: 4px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS['bg_dark']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border']};
        border-radius: 2px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['text_muted']};
    }}

    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 15px rgba(0, 240, 144, 0.3); }}
        50% {{ box-shadow: 0 0 30px rgba(0, 240, 144, 0.5); }}
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-3px); }}
    }}

    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}

    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-10px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    @keyframes ringProgress {{
        from {{ stroke-dashoffset: 175; }}
        to {{ stroke-dashoffset: var(--ring-offset); }}
    }}

    /* ===== COMMAND BAR (HEADER) ===== */
    .command-bar {{
        height: 60px;
        background: rgba(5, 5, 5, 0.9);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid {COLORS['border']};
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        position: sticky;
        top: 0;
        z-index: 100;
    }}

    .logo-section {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .logo-dot {{
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: linear-gradient(135deg, {COLORS['accent_green']}, {COLORS['accent_cyan']});
        animation: pulse 2s infinite;
    }}

    .logo-text {{
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: {COLORS['text_primary']};
    }}

    .logo-text .muted {{
        color: {COLORS['text_muted']};
    }}

    .api-status {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: {COLORS['text_muted']};
    }}

    .status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {COLORS['accent_green']};
        animation: pulse 2s infinite;
    }}

    /* ===== MAIN CONTENT ===== */
    .main-content {{
        display: flex;
        min-height: calc(100vh - 60px);
    }}

    /* ===== LEFT SIDEBAR (Intelligence Hub) ===== */
    .intelligence-hub {{
        width: 280px;
        min-width: 280px;
        background: {COLORS['bg_dark']};
        border-right: 1px solid {COLORS['border']};
        padding: 20px;
        overflow-y: auto;
    }}

    .section-title {{
        font-size: 11px;
        font-weight: 700;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 16px;
    }}

    /* ===== CENTER (Market Stage) ===== */
    .market-stage {{
        flex: 1;
        background: {COLORS['bg_dark']};
        padding: 24px 32px;
        position: relative;
    }}

    /* ===== RIGHT SIDEBAR (Financial Bento) ===== */
    .financial-bento {{
        width: 320px;
        min-width: 320px;
        background: {COLORS['bg_dark']};
        border-left: 1px solid {COLORS['border']};
        padding: 24px;
        overflow-y: auto;
    }}

    /* ===== TICKER HEADER ===== */
    .ticker-header {{
        margin-bottom: 24px;
        animation: fadeIn 0.5s ease-out;
    }}

    .ticker-symbol {{
        font-size: 48px;
        font-weight: 800;
        color: {COLORS['text_primary']};
        letter-spacing: -2px;
        margin: 0;
        line-height: 1;
    }}

    .ticker-price {{
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-top: 8px;
    }}

    .price-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    .price-change {{
        display: flex;
        align-items: center;
        gap: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 500;
    }}

    .price-change.positive {{
        color: {COLORS['accent_green']};
    }}

    .price-change.negative {{
        color: {COLORS['accent_red']};
    }}

    /* ===== TIMEFRAME BUTTONS ===== */
    .timeframe-buttons {{
        display: flex;
        gap: 8px;
        margin-top: 12px;
    }}

    .tf-btn {{
        padding: 6px 12px;
        font-size: 10px;
        font-weight: 700;
        border-radius: 6px;
        background: transparent;
        color: {COLORS['text_muted']};
        border: none;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .tf-btn:hover {{
        color: {COLORS['text_primary']};
    }}

    .tf-btn.active {{
        background: {COLORS['chart_grid']};
        color: {COLORS['text_primary']};
    }}

    /* ===== CONFIDENCE RING ===== */
    .confidence-container {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 24px;
    }}

    .confidence-ring {{
        position: relative;
        width: 64px;
        height: 64px;
    }}

    .ring-svg {{
        width: 100%;
        height: 100%;
        transform: rotate(-90deg);
    }}

    .ring-bg {{
        fill: none;
        stroke: {COLORS['chart_grid']};
        stroke-width: 4;
    }}

    .ring-progress {{
        fill: none;
        stroke: {COLORS['accent_green']};
        stroke-width: 4;
        stroke-linecap: round;
        stroke-dasharray: 175;
        stroke-dashoffset: var(--ring-offset, 175);
        transition: stroke-dashoffset 1s ease-out;
    }}

    .ring-progress.purple {{
        stroke: {COLORS['accent_purple']};
    }}

    .ring-value {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 12px;
        font-weight: 700;
        color: {COLORS['text_primary']};
    }}

    .verdict-text {{
        font-size: 14px;
        font-weight: 700;
        color: {COLORS['accent_green']};
    }}

    .verdict-text.sell {{
        color: {COLORS['accent_red']};
    }}

    .verdict-description {{
        font-size: 10px;
        color: {COLORS['text_muted']};
        line-height: 1.4;
        margin-top: 4px;
    }}

    /* ===== ANOMALY CARDS ===== */
    .anomaly-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
        cursor: default;
    }}

    .anomaly-card:hover {{
        border-color: {COLORS['text_muted']};
    }}

    .anomaly-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }}

    .anomaly-label {{
        font-size: 11px;
        color: {COLORS['text_muted']};
    }}

    .anomaly-icon {{
        font-size: 12px;
    }}

    .anomaly-value {{
        font-size: 14px;
        font-weight: 500;
    }}

    .anomaly-value.positive {{
        color: {COLORS['accent_green']};
    }}

    .anomaly-value.negative {{
        color: {COLORS['accent_red']};
    }}

    .anomaly-value.neutral {{
        color: {COLORS['text_primary']};
    }}

    /* ===== BENTO CARDS ===== */
    .bento-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}

    .bento-card:hover {{
        border-color: {COLORS['text_muted']};
    }}

    .bento-label {{
        font-size: 12px;
        color: {COLORS['text_muted']};
        margin-bottom: 4px;
    }}

    .bento-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin-bottom: 8px;
    }}

    .bento-sublabel {{
        font-size: 11px;
        color: {COLORS['text_muted']};
    }}

    .bento-progress {{
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: {COLORS['border']};
    }}

    .bento-progress-fill {{
        height: 100%;
        background: {COLORS['accent_green']};
        transition: width 1s ease-out;
    }}

    /* ===== BEAT BADGE ===== */
    .beat-badge {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        background: rgba(0, 240, 144, 0.1);
        border: 1px solid rgba(0, 240, 144, 0.3);
        color: {COLORS['accent_green']};
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-radius: 4px;
        animation: glow 2s infinite;
    }}

    .miss-badge {{
        background: rgba(255, 46, 80, 0.1);
        border-color: rgba(255, 46, 80, 0.3);
        color: {COLORS['accent_red']};
        animation: none;
    }}

    /* ===== INSIDER BAR ===== */
    .insider-bar-container {{
        margin: 12px 0;
    }}

    .insider-labels {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }}

    .insider-label {{
        font-size: 12px;
        font-weight: 600;
    }}

    .insider-label.sell {{
        color: {COLORS['accent_red']};
    }}

    .insider-label.buy {{
        color: {COLORS['accent_green']};
    }}

    .insider-bar {{
        display: flex;
        height: 8px;
        background: {COLORS['chart_grid']};
        border-radius: 4px;
        overflow: hidden;
    }}

    .insider-sell {{
        background: {COLORS['accent_red']};
        transition: width 0.5s ease-out;
    }}

    .insider-buy {{
        background: {COLORS['accent_green']};
        transition: width 0.5s ease-out;
    }}

    .insider-values {{
        display: flex;
        justify-content: space-between;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: {COLORS['text_muted']};
    }}

    /* ===== PRICE TARGET GAUGE ===== */
    .target-gauge {{
        position: relative;
        padding: 16px 0;
    }}

    .target-track {{
        width: 100%;
        height: 8px;
        background: {COLORS['chart_grid']};
        border-radius: 4px;
        position: relative;
    }}

    .target-marker {{
        position: absolute;
        top: -8px;
        transform: translateX(-50%);
    }}

    .marker-line {{
        width: 2px;
        height: 24px;
        margin: 0 auto 4px;
    }}

    .marker-line.current {{
        background: {COLORS['text_primary']};
    }}

    .marker-line.target {{
        background: {COLORS['accent_green']};
    }}

    .marker-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        text-align: center;
        white-space: nowrap;
    }}

    .marker-label.current {{
        background: {COLORS['border']};
        padding: 2px 6px;
        border-radius: 4px;
        color: {COLORS['text_primary']};
    }}

    .marker-label.target {{
        color: {COLORS['accent_green']};
        text-shadow: 0 0 10px rgba(0, 240, 144, 0.3);
    }}

    /* ===== SEARCH INPUT ===== */
    .stTextInput > div > div > input {{
        background: {COLORS['bg_card']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
        color: {COLORS['text_primary']} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        height: auto !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease !important;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {COLORS['accent_green']} !important;
        box-shadow: 0 0 0 3px rgba(0, 240, 144, 0.15) !important;
    }}

    .stTextInput > div > div > input::placeholder {{
        color: {COLORS['text_muted']} !important;
        text-transform: uppercase !important;
    }}

    /* ===== BUTTONS ===== */
    .stButton > button {{
        background: {COLORS['gradient_primary']} !important;
        border: none !important;
        border-radius: 8px !important;
        color: {COLORS['bg_dark']} !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 12px 24px !important;
        height: auto !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 240, 144, 0.3) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ===== CHECKBOX ===== */
    .stCheckbox > label {{
        color: {COLORS['text_muted']} !important;
        font-size: 12px !important;
    }}

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS['bg_card']};
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
        border: 1px solid {COLORS['border']};
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 6px;
        color: {COLORS['text_muted']};
        font-weight: 500;
        font-size: 13px;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['text_primary']};
        background: {COLORS['border']};
    }}

    .stTabs [aria-selected="true"] {{
        background: {COLORS['accent_green']} !important;
        color: {COLORS['bg_dark']} !important;
        font-weight: 700 !important;
    }}

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ===== NEWS CARDS ===== */
    .news-card {{
        background: {COLORS['bg_card']};
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        border-left: 3px solid {COLORS['border']};
        transition: all 0.2s ease;
        animation: slideIn 0.3s ease-out;
    }}

    .news-card:hover {{
        transform: translateX(4px);
        background: {COLORS['bg_card_hover']};
    }}

    .news-card.bullish {{
        border-left-color: {COLORS['accent_green']};
    }}

    .news-card.bearish {{
        border-left-color: {COLORS['accent_red']};
    }}

    .news-card.neutral {{
        border-left-color: {COLORS['accent_yellow']};
    }}

    .news-title {{
        font-size: 14px;
        font-weight: 500;
        color: {COLORS['text_primary']};
        margin-bottom: 8px;
        line-height: 1.4;
    }}

    .news-meta {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
    }}

    .news-source {{
        color: {COLORS['text_muted']};
    }}

    .news-sentiment {{
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
    }}

    .news-sentiment.bullish {{
        background: rgba(0, 240, 144, 0.15);
        color: {COLORS['accent_green']};
    }}

    .news-sentiment.bearish {{
        background: rgba(255, 46, 80, 0.15);
        color: {COLORS['accent_red']};
    }}

    .news-sentiment.neutral {{
        background: rgba(245, 158, 11, 0.15);
        color: {COLORS['accent_yellow']};
    }}

    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        color: {COLORS['text_primary']} !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 12px !important;
        color: {COLORS['text_muted']} !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}

    /* ===== DATA TABLE ===== */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid {COLORS['border']};
    }}

    .stDataFrame [data-testid="stTable"] {{
        background: {COLORS['bg_card']};
    }}

    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div {{
        background: {COLORS['border']} !important;
        border-radius: 4px;
    }}

    .stProgress > div > div > div {{
        background: {COLORS['gradient_primary']} !important;
        border-radius: 4px;
    }}

    /* ===== LOADING SPINNER ===== */
    .loading-overlay {{
        position: fixed;
        inset: 0;
        z-index: 200;
        background: rgba(5, 5, 5, 0.9);
        backdrop-filter: blur(8px);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}

    .loading-spinner {{
        width: 64px;
        height: 64px;
        border: 4px solid {COLORS['border']};
        border-top-color: {COLORS['accent_green']};
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 16px;
    }}

    .loading-text {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        color: {COLORS['accent_green']};
        animation: pulse 1.5s infinite;
    }}

    /* ===== DISCLAIMER ===== */
    .disclaimer-footer {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 16px 24px;
        margin-top: 24px;
        font-size: 11px;
        color: {COLORS['text_muted']};
        text-align: center;
    }}

    /* ===== SYSTEM STATUS ===== */
    .system-status {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: {COLORS['text_muted']};
        text-align: center;
        padding: 16px;
        margin-top: auto;
    }}

    /* ===== RADAR CHART ANNOTATION ===== */
    .radar-score {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        pointer-events: none;
    }}

    .radar-score-value {{
        font-size: 28px;
        font-weight: 700;
        color: {COLORS['text_primary']};
    }}

    .radar-score-label {{
        font-size: 9px;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
    }}

    /* ===== PREDICTION CARD ===== */
    .prediction-card {{
        background: {COLORS['bg_card']};
        border: 2px solid {COLORS['accent_green']};
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
        animation: fadeIn 0.5s ease-out;
    }}

    .prediction-card.risk {{
        border-color: {COLORS['accent_red']};
    }}

    .prediction-card.stagnation {{
        border-color: {COLORS['accent_yellow']};
    }}

    .prediction-icon {{
        font-size: 48px;
        margin-bottom: 8px;
        animation: float 2s ease-in-out infinite;
    }}

    .prediction-label {{
        font-size: 24px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 8px 0;
    }}

    .prediction-label.growth {{
        color: {COLORS['accent_green']};
    }}

    .prediction-label.risk {{
        color: {COLORS['accent_red']};
    }}

    .prediction-label.stagnation {{
        color: {COLORS['accent_yellow']};
    }}

    .prediction-confidence {{
        font-size: 14px;
        color: {COLORS['text_muted']};
    }}

    /* ===== GOLDEN TRIANGLE ===== */
    .golden-triangle {{
        display: flex;
        justify-content: space-around;
        gap: 16px;
        margin: 24px 0;
    }}

    .triangle-metric {{
        flex: 1;
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }}

    .triangle-metric:hover {{
        border-color: {COLORS['text_muted']};
        transform: translateY(-2px);
    }}

    .triangle-icon {{
        font-size: 24px;
        margin-bottom: 8px;
    }}

    .triangle-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        color: {COLORS['text_primary']};
    }}

    .triangle-label {{
        font-size: 12px;
        color: {COLORS['text_muted']};
        margin-top: 4px;
    }}

    .triangle-weight {{
        font-size: 11px;
        color: {COLORS['text_dark']};
        margin-top: 2px;
    }}

    .triangle-bar {{
        height: 4px;
        background: {COLORS['border']};
        border-radius: 2px;
        margin-top: 12px;
        overflow: hidden;
    }}

    .triangle-bar-fill {{
        height: 100%;
        border-radius: 2px;
        transition: width 1s ease-out;
    }}

    .triangle-bar-fill.green {{
        background: {COLORS['accent_green']};
    }}

    .triangle-bar-fill.purple {{
        background: {COLORS['accent_purple']};
    }}

    .triangle-bar-fill.cyan {{
        background: {COLORS['accent_cyan']};
    }}

    /* ===== RESPONSIVE ===== */
    @media (max-width: 1200px) {{
        .intelligence-hub,
        .financial-bento {{
            display: none;
        }}

        .market-stage {{
            padding: 16px;
        }}
    }}

    @media (max-width: 768px) {{
        .ticker-symbol {{
            font-size: 32px;
        }}

        .price-value {{
            font-size: 20px;
        }}

        .golden-triangle {{
            flex-direction: column;
        }}

        .triangle-value {{
            font-size: 24px;
        }}
    }}
</style>
"""

# =============================================================================
# PLOTLY CONFIGURATION
# =============================================================================

def get_plotly_layout(title: str = "", height: int = 400, show_legend: bool = True) -> dict:
    """Get standard Plotly layout configuration for dark theme."""
    return {
        'paper_bgcolor': COLORS['bg_card'],
        'plot_bgcolor': COLORS['bg_card'],
        'font': {'color': COLORS['text_primary'], 'family': 'Inter, sans-serif'},
        'title': {
            'text': title,
            'font': {'size': 14, 'color': COLORS['text_primary'], 'weight': 600},
            'x': 0.02,
            'xanchor': 'left'
        } if title else None,
        'height': height,
        'margin': {'l': 50, 'r': 20, 't': 40 if title else 20, 'b': 40},
        'showlegend': show_legend,
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': COLORS['text_secondary'], 'size': 11}
        },
        'xaxis': {
            'gridcolor': COLORS['chart_grid'],
            'zerolinecolor': COLORS['border'],
            'tickfont': {'color': COLORS['text_muted'], 'size': 10},
            'linecolor': COLORS['border']
        },
        'yaxis': {
            'gridcolor': COLORS['chart_grid'],
            'zerolinecolor': COLORS['border'],
            'tickfont': {'color': COLORS['text_muted'], 'size': 10},
            'linecolor': COLORS['border'],
            'side': 'right'
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': COLORS['bg_dark'],
            'bordercolor': COLORS['border'],
            'font': {'color': COLORS['text_primary'], 'size': 12}
        }
    }


def get_candlestick_colors() -> dict:
    """Get colors for candlestick charts."""
    return {
        'increasing_line_color': COLORS['accent_green'],
        'increasing_fillcolor': COLORS['accent_green'],
        'decreasing_line_color': COLORS['accent_red'],
        'decreasing_fillcolor': COLORS['accent_red']
    }


def get_radar_layout(height: int = 250) -> dict:
    """Get layout for radar charts."""
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': COLORS['text_muted'], 'size': 10},
        'height': height,
        'margin': {'l': 40, 'r': 40, 't': 20, 'b': 20},
        'showlegend': False,
        'polar': {
            'bgcolor': 'rgba(0,0,0,0)',
            'radialaxis': {
                'visible': True,
                'range': [0, 10],
                'gridcolor': COLORS['border'],
                'tickfont': {'size': 8, 'color': COLORS['text_muted']}
            },
            'angularaxis': {
                'gridcolor': COLORS['border'],
                'linecolor': COLORS['border']
            }
        }
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_number(value: float, decimals: int = 2) -> str:
    """Format large numbers with K, M, B suffixes."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def format_percent(value: float, include_sign: bool = True) -> str:
    """Format percentage with optional sign."""
    if include_sign and value > 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def get_sentiment_color(value: float) -> str:
    """Get color based on sentiment value."""
    if value > 0.1:
        return COLORS['accent_green']
    elif value < -0.1:
        return COLORS['accent_red']
    return COLORS['accent_yellow']


def get_prediction_class(prediction: str) -> str:
    """Get CSS class for prediction type."""
    return {
        "Growth": "growth",
        "Risk": "risk",
        "Stagnation": "stagnation"
    }.get(prediction, "stagnation")


def get_prediction_color(prediction: str) -> str:
    """Get color for prediction type."""
    return {
        "Growth": COLORS['accent_green'],
        "Risk": COLORS['accent_red'],
        "Stagnation": COLORS['accent_yellow']
    }.get(prediction, COLORS['accent_yellow'])


def get_prediction_icon(prediction: str) -> str:
    """Get icon for prediction type."""
    return {
        "Growth": "rocket",
        "Risk": "trending_down",
        "Stagnation": "trending_flat"
    }.get(prediction, "trending_flat")


# =============================================================================
# LEGACY CSS VARIABLES (for backward compatibility)
# =============================================================================

# These are kept for backward compatibility with other modules
PAGE_CSS = MASTER_CSS
CARD_CSS = ""
METRIC_CSS = ""
SENTIMENT_CSS = ""
MARKET_STATUS_CSS = ""
HEADER_CSS = ""
SIDEBAR_NAV_CSS = ""


def get_sparkline_layout(height: int = 30) -> dict:
    """Get layout for sparkline mini-charts."""
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'height': height,
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
        'showlegend': False,
        'xaxis': {'visible': False},
        'yaxis': {'visible': False}
    }


def get_change_color(value: float) -> str:
    """Get color based on positive/negative value."""
    if value > 0:
        return COLORS['accent_green']
    elif value < 0:
        return COLORS['accent_red']
    return COLORS['text_muted']


def get_sentiment_class(sentiment: str) -> str:
    """Get CSS class for sentiment type."""
    return {
        "bullish": "bullish",
        "bearish": "bearish",
        "neutral": "neutral"
    }.get(sentiment.lower(), "neutral")


def inject_css(css_string: str):
    """Inject CSS into Streamlit page."""
    import streamlit as st
    st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)


def get_all_css() -> str:
    """Get all CSS combined."""
    return MASTER_CSS
