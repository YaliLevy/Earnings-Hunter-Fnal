import { SlotNumber } from '../SlotNumber';
import type { FinancialSummary } from '../../types';

interface FinancialsCardProps {
  data: FinancialSummary | null;
}

export function FinancialsCard({ data }: FinancialsCardProps) {
  if (!data) {
    return <FinancialsCardEmpty />;
  }

  const epsActual = data.eps_actual ?? 0;
  const epsEstimated = data.eps_estimated ?? 0;
  const epsSurprise = data.eps_surprise ?? 0;
  const revActual = data.revenue_actual ?? 0;
  const revEstimated = data.revenue_estimated ?? 0;

  const isEpsBeat = epsSurprise >= 0;

  // Format revenue to billions
  const formatRevenue = (rev: number) => {
    if (rev >= 1e9) return `$${(rev / 1e9).toFixed(2)}B`;
    if (rev >= 1e6) return `$${(rev / 1e6).toFixed(2)}M`;
    return `$${rev.toFixed(2)}`;
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      {/* Header */}
      <h3 className="text-xs tracking-widest text-text-muted mb-4">FINANCIALS</h3>

      {/* EPS Data */}
      <div className="space-y-3">
        {/* Actual EPS */}
        <div className="flex justify-between items-baseline">
          <span className="text-xs text-text-muted">ACTUAL</span>
          <SlotNumber
            value={epsActual}
            prefix="$"
            className={`text-lg font-mono font-semibold ${
              isEpsBeat ? 'text-signal-green animate-pulse-green' : 'text-crimson'
            }`}
          />
        </div>

        {/* Estimated EPS */}
        <div className="flex justify-between items-baseline">
          <span className="text-xs text-text-muted">EST.</span>
          <span className="text-lg font-mono text-text-muted">${epsEstimated.toFixed(2)}</span>
        </div>

        {/* Surprise */}
        <div className="flex justify-between items-baseline">
          <span className="text-xs text-text-muted">SURPRISE</span>
          <span
            className={`text-lg font-mono font-bold ${
              isEpsBeat ? 'text-signal-green' : 'text-crimson'
            }`}
          >
            {isEpsBeat ? '+' : ''}
            {epsSurprise.toFixed(2)}%
          </span>
        </div>

        {/* Divider */}
        <div className="h-px bg-border my-4" />

        {/* Revenue */}
        <div className="flex justify-between items-baseline">
          <span className="text-xs text-text-muted">REVENUE</span>
          <span className="text-base font-mono text-text-primary">
            {formatRevenue(revActual)}
          </span>
        </div>

        {/* Estimated Revenue */}
        <div className="flex justify-between items-baseline">
          <span className="text-xs text-text-muted">EST. REV</span>
          <span className="text-base font-mono text-text-muted">
            {formatRevenue(revEstimated)}
          </span>
        </div>
      </div>
    </div>
  );
}

function FinancialsCardEmpty() {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <h3 className="text-xs tracking-widest text-text-muted mb-4">FINANCIALS</h3>
      <div className="text-center py-6 text-text-muted text-sm">No data</div>
    </div>
  );
}
