import { useState, useEffect, useCallback } from 'react';
import { AnimatePresence } from 'framer-motion';

// Components
import { LegalDisclaimer } from './components/LegalDisclaimer';
import { CommandBar } from './components/CommandBar';
import { TickerStrip } from './components/TickerStrip';
import { IntelligenceHub } from './components/IntelligenceHub';
import { MarketStage } from './components/MarketStage';
import { FinancialIntel } from './components/FinancialIntel';
import { FooterDisclaimer } from './components/FooterDisclaimer';

// Services & Types
import { getAnalysis, getHistoricalData, getNews, runDeepAnalysis } from './services/api';
import type { AnalysisResult, HistoricalData, Timeframe } from './types';

function App() {
  // Disclaimer state
  const [showDisclaimer, setShowDisclaimer] = useState(() => {
    return localStorage.getItem('disclaimer_accepted') !== 'true';
  });

  // UI state
  const [deepReasoning, setDeepReasoning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data state
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalData | null>(null);
  const [timeframe, setTimeframe] = useState<Timeframe>('3M');
  const [deepInsight, setDeepInsight] = useState<string | null>(null);

  // Accept disclaimer
  const handleAcceptDisclaimer = () => {
    localStorage.setItem('disclaimer_accepted', 'true');
    setShowDisclaimer(false);
  };

  // Search handler
  const handleSearch = useCallback(async (symbol: string) => {
    if (!symbol.trim()) return;

    setIsLoading(true);
    setError(null);
    setDeepInsight(null);

    try {
      // Fetch analysis and historical data in parallel
      const [analysisResult, historyResult] = await Promise.all([
        getAnalysis(symbol),
        getHistoricalData(symbol, timeframe),
      ]);

      setAnalysis(analysisResult);
      setHistoricalData(historyResult);

      // Fetch news and update analysis with it
      try {
        const newsResult = await getNews(symbol);
        setAnalysis((prev) =>
          prev ? { ...prev, news_articles: newsResult } : prev
        );
      } catch (newsError) {
        console.warn('Failed to fetch news:', newsError);
      }

      // If deep reasoning is enabled, run it asynchronously
      if (deepReasoning) {
        runDeepAnalysis(symbol, analysisResult.prediction, analysisResult.confidence)
          .then((result) => {
            setDeepInsight(result.report);
          })
          .catch((err) => {
            console.warn('Deep analysis failed:', err);
          });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setError(message);
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [timeframe, deepReasoning]);

  // Timeframe change handler
  const handleTimeframeChange = useCallback(async (newTimeframe: Timeframe) => {
    setTimeframe(newTimeframe);

    if (analysis?.symbol) {
      try {
        const historyResult = await getHistoricalData(analysis.symbol, newTimeframe);
        setHistoricalData(historyResult);
      } catch (err) {
        console.error('Failed to fetch historical data:', err);
      }
    }
  }, [analysis?.symbol]);

  return (
    <div className="h-screen flex flex-col bg-bg-dark overflow-hidden">
      {/* Legal Disclaimer Modal */}
      <LegalDisclaimer open={showDisclaimer} onAccept={handleAcceptDisclaimer} />

      {/* Command Bar (Zone A) */}
      <CommandBar
        deepReasoning={deepReasoning}
        onDeepReasoningChange={setDeepReasoning}
        onSearch={handleSearch}
        isLoading={isLoading}
      />

      {/* Ticker Strip (Zone B) - Animated */}
      <AnimatePresence>
        {analysis && (
          <TickerStrip
            symbol={analysis.symbol}
            companyName={analysis.company_name}
            price={analysis.financial_summary.current_price || 0}
            change={analysis.financial_summary.price_change || 0}
            changePercent={analysis.financial_summary.price_change_pct || 0}
            volume={analysis.financial_summary.volume}
            marketCap={analysis.financial_summary.market_cap}
          />
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <div className="flex-1 flex min-h-0">
        {/* Intelligence Hub (Zone C - Left 20%) */}
        <div className="w-[260px] min-w-[220px] border-r border-border bg-bg-dark">
          <IntelligenceHub analysis={analysis} deepInsight={deepInsight} />
        </div>

        {/* Market Stage (Zone D - Center 60%) */}
        <div className="flex-1 bg-bg-dark">
          <MarketStage
            symbol={analysis?.symbol || null}
            historicalData={historicalData}
            timeframe={timeframe}
            onTimeframeChange={handleTimeframeChange}
            isLoading={isLoading}
          />
        </div>

        {/* Financial Intel (Zone E - Right 20%) */}
        <div className="w-[280px] min-w-[220px] border-l border-border bg-bg-dark">
          <FinancialIntel analysis={analysis} />
        </div>
      </div>

      {/* Error Toast */}
      {error && (
        <div className="fixed bottom-12 left-1/2 -translate-x-1/2 bg-crimson/90 text-white
                        px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-up">
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-4 text-white/80 hover:text-white"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Footer Disclaimer */}
      <FooterDisclaimer />
    </div>
  );
}

export default App;
