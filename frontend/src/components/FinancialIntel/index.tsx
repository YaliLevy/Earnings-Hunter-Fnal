import { FinancialsCard } from './FinancialsCard';
import { EarningsCallCard } from './EarningsCallCard';
import { NewsCard } from './NewsCard';
import { InsiderActivityCard } from './InsiderActivityCard';
import type { AnalysisResult } from '../../types';

interface FinancialIntelProps {
  analysis: AnalysisResult | null;
}

export function FinancialIntel({ analysis }: FinancialIntelProps) {
  if (!analysis) {
    return <FinancialIntelEmpty />;
  }

  return (
    <div className="h-full flex flex-col p-5 overflow-y-auto">
      {/* Zone Title */}
      <h2 className="text-xs tracking-widest text-text-muted mb-4">
        FINANCIAL INTEL
      </h2>

      {/* Cards Stack */}
      <div className="space-y-3">
        {/* Financials Card */}
        <FinancialsCard data={analysis.financial_summary} />

        {/* Earnings Call Card */}
        <EarningsCallCard
          data={analysis.ceo_tone_summary}
          symbol={analysis.symbol}
          quarterLabel={analysis.quarter_label}
          transcriptContent={analysis.transcript_content}
        />

        {/* News Card */}
        <NewsCard articles={analysis.news_articles} />

        {/* Insider Activity Card */}
        <InsiderActivityCard
          financialData={analysis.financial_summary}
          symbol={analysis.symbol}
          insiderTransactions={analysis.insider_transactions}
        />
      </div>
    </div>
  );
}

function FinancialIntelEmpty() {
  return (
    <div className="h-full flex flex-col p-5 overflow-y-auto">
      {/* Zone Title */}
      <h2 className="text-xs tracking-widest text-text-muted mb-4">
        FINANCIAL INTEL
      </h2>

      {/* Empty Cards */}
      <div className="space-y-3">
        {['FINANCIALS', 'EARNINGS CALL', 'LATEST NEWS', 'INSIDER ACTIVITY'].map((title) => (
          <div key={title} className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-xs tracking-widest text-text-muted mb-4">{title}</h3>
            <div className="text-center py-6 text-text-muted text-sm">No data</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export { FinancialsCard, EarningsCallCard, NewsCard, InsiderActivityCard };
