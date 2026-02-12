import { ConfidenceRing } from './ConfidenceRing';
import { GoldenTriangleRadar } from './GoldenTriangleRadar';
import { AIAnalysis } from './AIAnalysis';
import type { AnalysisResult } from '../../types';

interface IntelligenceHubProps {
  analysis: AnalysisResult | null;
  deepInsight?: string | null;
}

export function IntelligenceHub({ analysis, deepInsight }: IntelligenceHubProps) {
  if (!analysis) {
    return <IntelligenceHubEmpty />;
  }

  return (
    <div className="h-full flex flex-col p-5 overflow-y-auto">
      {/* Zone Title */}
      <h2 className="text-xs tracking-widest text-text-muted mb-4">
        INTELLIGENCE HUB
      </h2>

      {/* Confidence Ring */}
      <ConfidenceRing
        confidence={analysis.confidence}
        prediction={analysis.prediction}
      />

      {/* Divider */}
      <div className="h-px bg-border my-4" />

      {/* Golden Triangle Radar */}
      <GoldenTriangleRadar data={analysis.golden_triangle} />

      {/* Divider */}
      <div className="h-px bg-border my-4" />

      {/* AI Analysis */}
      <AIAnalysis
        symbol={analysis.symbol}
        companyName={analysis.company_name}
        prediction={analysis.prediction}
        confidence={analysis.confidence}
        epsBeat={analysis.financial_summary.eps_beat}
        epsSurprise={analysis.financial_summary.eps_surprise}
        ceoPhrases={analysis.ceo_tone_summary.key_positive_phrases || []}
        deepInsight={deepInsight}
      />
    </div>
  );
}

function IntelligenceHubEmpty() {
  return (
    <div className="h-full flex flex-col p-5">
      {/* Zone Title */}
      <h2 className="text-xs tracking-widest text-text-muted mb-4">
        INTELLIGENCE HUB
      </h2>

      {/* Empty Confidence Ring */}
      <div className="flex flex-col items-center py-4">
        <div className="relative" style={{ width: 180, height: 180 }}>
          <svg
            width={180}
            height={180}
            viewBox="0 0 180 180"
            className="transform -rotate-90"
          >
            <circle
              cx={90}
              cy={90}
              r={84}
              fill="none"
              stroke="hsl(220, 15%, 20%)"
              strokeWidth={12}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-5xl font-mono font-bold text-text-muted">
              --%
            </span>
            <span className="text-xs tracking-widest text-text-muted mt-1">
              CONFIDENCE
            </span>
          </div>
        </div>

        <div className="text-xl font-bold tracking-wider mt-4 text-text-muted">
          WAITING
        </div>
      </div>

      {/* Empty state message */}
      <div className="flex-1 flex items-center justify-center">
        <p className="text-sm text-text-muted text-center px-4">
          Enter a ticker symbol above to begin analysis
        </p>
      </div>
    </div>
  );
}

export { ConfidenceRing, GoldenTriangleRadar, AIAnalysis };
