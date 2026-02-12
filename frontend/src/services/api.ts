/**
 * API Service for Earnings Hunter
 *
 * Communicates with the FastAPI backend.
 */

import type {
  Quote,
  AnalysisResult,
  HistoricalData,
  NewsArticle,
  InsiderTrade,
  TranscriptData,
  Timeframe,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

/**
 * Safe fetch wrapper with error handling
 */
async function safeFetch<T>(url: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Network error');
  }
}

/**
 * Get real-time stock quote
 */
export async function getQuote(symbol: string): Promise<Quote> {
  return safeFetch<Quote>(`${API_BASE}/api/quote/${symbol.toUpperCase()}`);
}

/**
 * Get full analysis with Golden Triangle
 */
export async function getAnalysis(symbol: string, forceRefresh = false): Promise<AnalysisResult> {
  const params = new URLSearchParams();
  if (forceRefresh) params.set('force_refresh', 'true');

  const url = `${API_BASE}/api/analyze/${symbol.toUpperCase()}${params.toString() ? '?' + params : ''}`;
  return safeFetch<AnalysisResult>(url);
}

/**
 * Get historical price data for charts
 */
export async function getHistoricalData(
  symbol: string,
  timeframe: Timeframe = '3M'
): Promise<HistoricalData> {
  return safeFetch<HistoricalData>(
    `${API_BASE}/api/historical/${symbol.toUpperCase()}?timeframe=${timeframe}`
  );
}

/**
 * Get news articles with sentiment
 */
export async function getNews(symbol: string, limit = 20): Promise<NewsArticle[]> {
  return safeFetch<NewsArticle[]>(
    `${API_BASE}/api/news/${symbol.toUpperCase()}?limit=${limit}`
  );
}

/**
 * Get insider trading activity
 */
export async function getInsiderTrades(symbol: string, limit = 50): Promise<InsiderTrade[]> {
  return safeFetch<InsiderTrade[]>(
    `${API_BASE}/api/insiders/${symbol.toUpperCase()}?limit=${limit}`
  );
}

/**
 * Get earnings call transcript
 */
export async function getTranscript(
  symbol: string,
  year?: number,
  quarter?: number
): Promise<TranscriptData> {
  const params = new URLSearchParams();
  if (year) params.set('year', year.toString());
  if (quarter) params.set('quarter', quarter.toString());

  const url = `${API_BASE}/api/transcript/${symbol.toUpperCase()}${params.toString() ? '?' + params : ''}`;
  return safeFetch<TranscriptData>(url);
}

/**
 * Run deep analysis with CrewAI agents
 */
export async function runDeepAnalysis(
  symbol: string,
  prediction: string,
  confidence: number
): Promise<{ symbol: string; report: string; status: string }> {
  const params = new URLSearchParams({
    prediction,
    confidence: confidence.toString(),
  });

  return safeFetch<{ symbol: string; report: string; status: string }>(
    `${API_BASE}/api/deep-analysis/${symbol.toUpperCase()}?${params}`,
    { method: 'POST' }
  );
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{ status: string }> {
  return safeFetch<{ status: string }>(`${API_BASE}/health`);
}
