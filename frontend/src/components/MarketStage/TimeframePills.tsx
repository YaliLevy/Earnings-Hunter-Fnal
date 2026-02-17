import type { Timeframe } from '../../types';

interface TimeframePillsProps {
  selected: Timeframe;
  onChange: (timeframe: Timeframe) => void;
  disabled?: boolean;
}

const TIMEFRAMES: Timeframe[] = ['1D', '1W', '1M', '3M', '6M', '1Y'];

export function TimeframePills({ selected, onChange, disabled }: TimeframePillsProps) {
  return (
    <div className="flex gap-2 justify-center">
      {TIMEFRAMES.map((tf) => (
        <button
          key={tf}
          onClick={() => onChange(tf)}
          disabled={disabled}
          className={`pill-button ${tf === selected ? 'active' : ''}`}
        >
          {tf}
        </button>
      ))}
    </div>
  );
}
