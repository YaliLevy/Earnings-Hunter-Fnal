import { motion } from 'framer-motion';
import { SlotNumber } from './SlotNumber';

interface TickerStripProps {
  symbol: string;
  companyName: string | null;
  price: number;
  change: number;
  changePercent: number;
  volume: number | null;
  marketCap: number | null;
}

export function TickerStrip({
  symbol,
  companyName,
  price,
  change,
  changePercent,
  volume,
  marketCap,
}: TickerStripProps) {
  const isPositive = change >= 0;

  // Format large numbers
  const formatVolume = (vol: number | null) => {
    if (!vol) return 'N/A';
    if (vol >= 1e9) return `${(vol / 1e9).toFixed(1)}B`;
    if (vol >= 1e6) return `${(vol / 1e6).toFixed(1)}M`;
    if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`;
    return vol.toString();
  };

  const formatMarketCap = (cap: number | null) => {
    if (!cap) return 'N/A';
    if (cap >= 1e12) return `$${(cap / 1e12).toFixed(1)}T`;
    if (cap >= 1e9) return `$${(cap / 1e9).toFixed(1)}B`;
    if (cap >= 1e6) return `$${(cap / 1e6).toFixed(1)}M`;
    return `$${cap}`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 44 }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="bg-bg-dark border-b border-border px-6 flex items-center justify-between overflow-hidden"
    >
      {/* Left: Company info and price */}
      <div className="flex items-center gap-5">
        <span className="text-sm font-semibold text-text-primary">
          {companyName || symbol}
        </span>

        <div className="flex items-baseline gap-3">
          <SlotNumber
            value={price}
            prefix="$"
            className="text-xl font-mono font-semibold text-text-primary"
          />

          <span
            className={`text-sm font-mono font-medium ${
              isPositive ? 'text-signal-green' : 'text-crimson'
            }`}
          >
            {isPositive ? '+' : ''}
            {change.toFixed(2)} ({isPositive ? '+' : ''}
            {changePercent.toFixed(2)}%)
          </span>
        </div>
      </div>

      {/* Right: Volume and Market Cap */}
      <div className="flex items-center gap-6 text-xs font-mono text-text-muted">
        <span>Vol: {formatVolume(volume)}</span>
        <span>MCap: {formatMarketCap(marketCap)}</span>
      </div>
    </motion.div>
  );
}
