import * as Dialog from '@radix-ui/react-dialog';
import { AlertTriangle } from 'lucide-react';

interface LegalDisclaimerProps {
  open: boolean;
  onAccept: () => void;
}

export function LegalDisclaimer({ open, onAccept }: LegalDisclaimerProps) {
  return (
    <Dialog.Root open={open}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50" />
        <Dialog.Content
          className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                     bg-surface border border-border rounded-2xl p-8 max-w-2xl w-[90vw]
                     z-50 animate-fade-in [&>button]:hidden"
          onPointerDownOutside={(e) => e.preventDefault()}
          onEscapeKeyDown={(e) => e.preventDefault()}
        >
          <div className="flex flex-col items-center text-center">
            {/* Warning Icon */}
            <div className="w-16 h-16 rounded-full bg-amber/10 flex items-center justify-center mb-6">
              <AlertTriangle className="w-8 h-8 text-amber" />
            </div>

            {/* Title */}
            <Dialog.Title className="text-xl font-bold tracking-wider text-text-primary mb-6">
              LEGAL DISCLAIMER
            </Dialog.Title>

            {/* Content */}
            <div className="space-y-4 text-sm text-text-secondary leading-relaxed mb-8">
              <p>
                The information provided on this platform is for{' '}
                <span className="text-text-primary font-semibold">
                  informational and educational purposes only
                </span>{' '}
                and does not constitute financial advice, investment recommendations,
                or an offer to buy or sell any securities.
              </p>

              <p>
                All analysis, including AI-generated insights, Golden Triangle scores,
                and predictions, are based on historical data and algorithmic models.{' '}
                <span className="text-crimson font-semibold">
                  Past performance is not indicative of future results.
                </span>{' '}
                Markets are inherently unpredictable, and any investment carries risk of loss.
              </p>

              <p>
                Always conduct your own research and consult with a{' '}
                <span className="text-text-primary font-semibold">
                  licensed financial advisor
                </span>{' '}
                before making any investment decisions. By using this platform, you acknowledge
                that you understand and accept these risks.
              </p>
            </div>

            {/* Accept Button */}
            <button
              onClick={onAccept}
              className="neon-green-button text-sm tracking-wider font-semibold"
            >
              I Understand & Accept
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
