import { PriceChart } from './PriceChart';
import { TimeframePills } from './TimeframePills';
import type { HistoricalData, Timeframe } from '../../types';

interface MarketStageProps {
  symbol: string | null;
  historicalData: HistoricalData | null;
  timeframe: Timeframe;
  onTimeframeChange: (tf: Timeframe) => void;
  isLoading: boolean;
}

export function MarketStage({
  symbol,
  historicalData,
  timeframe,
  onTimeframeChange,
  isLoading,
}: MarketStageProps) {
  return (
    <div className="h-full flex flex-col p-5">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-baseline gap-3">
          <h2 className="text-sm font-medium text-text-primary">
            {symbol || 'AAPL'} — {timeframe}
          </h2>
          <span className="text-xs text-text-muted">SMA(20) + Projection</span>
        </div>

        {/* Timeframe Pills */}
        <TimeframePills
          selected={timeframe}
          onChange={onTimeframeChange}
          disabled={isLoading || !symbol}
        />
      </div>

      {/* Chart Area */}
      <div className="flex-1 min-h-0 relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-bg-dark/50 z-10">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-signal-green border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-text-muted">Loading chart...</span>
            </div>
          </div>
        )}

        <PriceChart data={historicalData} symbol={symbol || ''} />
      </div>
    </div>
  );
}

export { PriceChart, TimeframePills };
