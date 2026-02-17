// API Response Types

export interface Quote {
  symbol: string;
  name: string | null;
  price: number;
  change: number | null;
  change_percent: number | null;
  day_low: number | null;
  day_high: number | null;
  year_low: number | null;
  year_high: number | null;
  market_cap: number | null;
  volume: number | null;
  avg_volume: number | null;
  open_price: number | null;
  previous_close: number | null;
  pe: number | null;
  eps: number | null;
}

export interface GoldenTriangleScore {
  score: number;
  weight: number;
  weighted_score: number;
  label: string;
}

export interface GoldenTriangle {
  financial: GoldenTriangleScore;
  ceo_tone: GoldenTriangleScore;
  social: GoldenTriangleScore;
  composite: {
    score: number;
    label: string;
  };
}

export interface FinancialSummary {
  eps_actual: number | null;
  eps_estimated: number | null;
  eps_surprise: number | null;
  eps_beat: boolean | null;
  revenue_actual: number | null;
  revenue_estimated: number | null;
  revenue_surprise: number | null;
  revenue_beat: boolean | null;
  double_beat: boolean | null;
  current_price: number | null;
  price_change: number | null;
  price_change_pct: number | null;
  market_cap: number | null;
  volume: number | null;
  insider_buys: number;
  insider_sells: number;
}

export interface CEOToneSummary {
  has_transcript: boolean;
  confidence_score: number | null;
  sentiment_score: number | null;
  tone_summary: string | null;
  executive_summary: string | null;
  key_positive_phrases: string[];
  key_negative_phrases: string[];
}

export interface SocialSummary {
  available: boolean;
  news_count: number;
  combined_sentiment: number;
  bullish_percentage: number;
  sentiment_classification: string;
}

export interface NewsArticle {
  title: string;
  url: string | null;
  published_date: string | null;
  source: string | null;
  score: number;
  sentiment: 'bullish' | 'bearish' | 'neutral';
}

export interface InsiderTrade {
  date: string;
  reporter: string;
  transaction: string;
  shares: number;
  price: number | null;
  value: number | null;
  type: 'BUY' | 'SELL' | 'OTHER';
}

export interface CompositeComponent {
  key: string;
  label: string;
  score: number;         // 0-100
  weight: number;        // percentage (e.g. 25)
  weighted_contribution: number;
  reasoning: string;
}

export interface CompositeBreakdown {
  composite_score: number;  // 0-100
  components: CompositeComponent[];
  overall_reasoning: string;
}

export interface MLConsensus {
  agreement_ratio: number;
  vote_distribution: Record<string, number>;
  models_agree: number;
  models_total: number;
  best_model_confidence: number | null;
  best_model_name: string;
}

export interface AnalysisResult {
  symbol: string;
  company_name: string | null;
  earnings_date: string;
  year: number;
  quarter: number;
  quarter_label: string;
  prediction: 'Growth' | 'Risk' | 'Stagnation';
  confidence: number;
  golden_triangle: GoldenTriangle;
  financial_summary: FinancialSummary;
  ceo_tone_summary: CEOToneSummary;
  social_summary: SocialSummary;
  news_articles: NewsArticle[];
  insider_transactions: InsiderTrade[];
  transcript_content: string | null;
  research_insight: string | null;
  disclaimer: string;
  composite_breakdown: CompositeBreakdown | null;
  ml_consensus: MLConsensus | null;
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
  change: number | null;
  change_percent: number | null;
}

export interface HistoricalData {
  symbol: string;
  timeframe: string;
  prices: PricePoint[];
  sma20: (number | null)[];
  forecast: PricePoint[];
}

export interface TranscriptData {
  symbol: string;
  year: number;
  quarter: number;
  date: string | null;
  content: string;
  analysis: {
    confidence_score: number | null;
    sentiment_score: number | null;
    tone_summary: string | null;
    key_positive_phrases: string[];
    key_negative_phrases: string[];
  };
}

// UI State Types
export type Timeframe = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y';

export interface AppState {
  showDisclaimer: boolean;
  deepReasoning: boolean;
  searchQuery: string;
  isLoading: boolean;
  error: string | null;
  analysis: AnalysisResult | null;
  historicalData: HistoricalData | null;
  timeframe: Timeframe;
}
