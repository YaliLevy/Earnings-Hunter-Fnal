import { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import * as ScrollArea from '@radix-ui/react-scroll-area';
import { motion } from 'framer-motion';
import { ChevronRight, X } from 'lucide-react';
import type { InsiderTrade, FinancialSummary } from '../../types';

interface InsiderActivityCardProps {
  financialData: FinancialSummary | null;
  symbol: string;
  insiderTransactions: InsiderTrade[];
}

export function InsiderActivityCard({ financialData, symbol, insiderTransactions }: InsiderActivityCardProps) {
  const [isOpen, setIsOpen] = useState(false);

  const buys = financialData?.insider_buys ?? 0;
  const sells = financialData?.insider_sells ?? 0;
  const total = buys + sells;

  const buyPercent = total > 0 ? (buys / total) * 100 : 50;
  const sellPercent = total > 0 ? (sells / total) * 100 : 50;

  const hasData = insiderTransactions && insiderTransactions.length > 0;

  return (
    <>
      {/* Card */}
      <div
        className={`bg-surface border border-border rounded-lg p-4 transition-all ${
          hasData ? 'cursor-pointer hover:border-signal-green group' : ''
        }`}
        onClick={() => hasData && setIsOpen(true)}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs tracking-widest text-text-muted">INSIDER ACTIVITY</h3>
          {hasData && (
            <ChevronRight className="w-4 h-4 text-text-muted group-hover:text-signal-green transition-colors" />
          )}
        </div>

        {/* Stacked Bar */}
        <div className="h-6 bg-border rounded overflow-hidden flex">
          <div
            className="bg-signal-green transition-all"
            style={{ width: `${buyPercent}%` }}
          />
          <div
            className="bg-crimson transition-all"
            style={{ width: `${sellPercent}%` }}
          />
        </div>

        {/* Counts */}
        <div className="flex justify-between mt-3 font-mono text-xs">
          <span className="text-signal-green">BUY: {buys}</span>
          <span className="text-crimson">SELL: {sells}</span>
        </div>
      </div>

      {/* Dialog */}
      <Dialog.Root open={isOpen} onOpenChange={setIsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50" />
          <Dialog.Content
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                       bg-surface border border-border rounded-2xl max-w-2xl w-[90vw] max-h-[80vh]
                       z-50 flex flex-col overflow-hidden"
          >
            {/* Dialog Header */}
            <div className="flex items-center justify-between p-6 border-b border-border">
              <Dialog.Title className="text-lg font-semibold tracking-wider">
                INSIDER ACTIVITY — {symbol}
              </Dialog.Title>
              <Dialog.Close asChild>
                <button className="text-text-muted hover:text-text-primary transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </Dialog.Close>
            </div>

            {/* Summary */}
            <div className="mx-6 mt-6 flex gap-4">
              <div className="flex-1 bg-signal-green/10 border border-signal-green/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-mono font-bold text-signal-green">{buys}</div>
                <div className="text-xs text-text-muted">BUY TRANSACTIONS</div>
              </div>
              <div className="flex-1 bg-crimson/10 border border-crimson/30 rounded-lg p-4 text-center">
                <div className="text-2xl font-mono font-bold text-crimson">{sells}</div>
                <div className="text-xs text-text-muted">SELL TRANSACTIONS</div>
              </div>
            </div>

            {/* Trades List */}
            <ScrollArea.Root className="flex-1 overflow-hidden m-6">
              <ScrollArea.Viewport className="w-full h-full">
                {insiderTransactions && insiderTransactions.length > 0 ? (
                  <div className="space-y-3">
                    {insiderTransactions.map((trade, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.03 }}
                        className="bg-surface/50 border border-border rounded-lg p-3"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <div className="font-semibold text-text-primary text-sm">
                              {trade.reporter}
                            </div>
                            <div className="text-xs text-text-muted">{trade.date}</div>
                          </div>
                          <span
                            className={`badge ${
                              trade.type === 'BUY' ? 'badge-bullish' : 'badge-bearish'
                            }`}
                          >
                            {trade.type}
                          </span>
                        </div>
                        <div className="flex gap-4 text-xs font-mono text-text-muted">
                          <span>{trade.shares.toLocaleString()} shares</span>
                          {trade.price && <span>${trade.price.toFixed(2)}</span>}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-text-muted">
                    No insider trades found
                  </div>
                )}
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
