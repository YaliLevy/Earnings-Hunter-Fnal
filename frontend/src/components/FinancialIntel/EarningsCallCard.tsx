import { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import * as ScrollArea from '@radix-ui/react-scroll-area';
import { ChevronRight, X } from 'lucide-react';
import type { CEOToneSummary } from '../../types';

interface EarningsCallCardProps {
  data: CEOToneSummary | null;
  symbol: string;
  quarterLabel: string;
  transcriptContent: string | null;
}

export function EarningsCallCard({
  data,
  symbol,
  quarterLabel,
  transcriptContent,
}: EarningsCallCardProps) {
  const [isOpen, setIsOpen] = useState(false);

  const hasTranscript = data?.has_transcript ?? false;
  const confidenceScore = data?.confidence_score ?? 0;
  const sentimentScore = data?.sentiment_score ?? 0.5;
  const summary = data?.executive_summary || 'Click to view transcript...';

  // Determine sentiment badge
  const getBadge = () => {
    if (!hasTranscript) return { text: 'NO DATA', class: 'badge-neutral' };
    if (sentimentScore > 0.6) return { text: 'BULLISH', class: 'badge-bullish' };
    if (sentimentScore < 0.4) return { text: 'BEARISH', class: 'badge-bearish' };
    return { text: 'NEUTRAL', class: 'badge-neutral' };
  };

  const badge = getBadge();

  return (
    <>
      {/* Card */}
      <div
        className={`bg-surface border border-border rounded-lg p-4 transition-all ${
          hasTranscript ? 'cursor-pointer hover:border-signal-green group' : ''
        }`}
        onClick={() => hasTranscript && setIsOpen(true)}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs tracking-widest text-text-muted">EARNINGS CALL</h3>
          {hasTranscript && (
            <ChevronRight className="w-4 h-4 text-text-muted group-hover:text-signal-green transition-colors" />
          )}
        </div>

        {/* Quarter */}
        <div className="font-mono text-xs text-text-muted mb-2">{quarterLabel}</div>

        {/* Badge and Confidence */}
        <div className="flex items-center gap-2 mb-2">
          <span className={`badge ${badge.class}`}>{badge.text}</span>
          {hasTranscript && (
            <span className="text-xs text-text-muted">
              {(confidenceScore * 100).toFixed(0)}%
            </span>
          )}
        </div>

        {/* Summary preview */}
        <p className="text-sm text-text-muted line-clamp-2">
          {summary.slice(0, 100)}
          {summary.length > 100 ? '...' : ''}
        </p>
      </div>

      {/* Dialog */}
      <Dialog.Root open={isOpen} onOpenChange={setIsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50" />
          <Dialog.Content
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                       bg-surface border border-border rounded-2xl max-w-4xl w-[90vw] max-h-[80vh]
                       z-50 flex flex-col overflow-hidden"
          >
            {/* Dialog Header */}
            <div className="flex items-center justify-between p-6 border-b border-border">
              <Dialog.Title className="text-lg font-semibold tracking-wider">
                EARNINGS CALL — {quarterLabel}
              </Dialog.Title>
              <Dialog.Close asChild>
                <button className="text-text-muted hover:text-text-primary transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </Dialog.Close>
            </div>

            {/* Summary Card */}
            <div className="mx-6 mt-6 p-4 bg-signal-green/10 border border-signal-green/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className={`badge ${badge.class}`}>{badge.text}</span>
                <span className="text-sm text-text-secondary">
                  Confidence: {(confidenceScore * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-text-secondary leading-relaxed">{summary}</p>
            </div>

            {/* Transcript */}
            <ScrollArea.Root className="flex-1 overflow-hidden m-6">
              <ScrollArea.Viewport className="w-full h-full">
                <pre className="font-mono text-xs text-text-muted whitespace-pre-wrap leading-relaxed">
                  {transcriptContent || 'No transcript available.'}
                </pre>
              </ScrollArea.Viewport>
              <ScrollArea.Scrollbar
                className="flex w-2 p-0.5 bg-transparent"
                orientation="vertical"
              >
                <ScrollArea.Thumb className="relative flex-1 bg-border rounded-full" />
              </ScrollArea.Scrollbar>
            </ScrollArea.Root>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
}
