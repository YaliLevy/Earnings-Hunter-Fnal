import { AlertTriangle } from 'lucide-react';

export function FooterDisclaimer() {
  return (
    <div className="fixed bottom-0 left-0 right-0 h-9 bg-bg-dark border-t border-border
                    flex items-center px-6 z-30">
      <div className="flex items-center gap-2 text-xs text-text-muted">
        <AlertTriangle className="w-3 h-3 text-amber shrink-0" />
        <span className="tracking-wide truncate">
          Disclaimer: The information provided on this platform is for informational
          and educational purposes only and does not constitute financial advice,
          investment recommendations, or an offer to buy or sell any securities.
          Always consult a licensed financial advisor before making investment decisions.
          Past performance is not indicative of future results.
        </span>
      </div>
    </div>
  );
}
