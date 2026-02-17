import { useEffect, useState, memo } from 'react';

interface SlotNumberProps {
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  className?: string;
}

/**
 * SlotNumber - Animated digit rolling number display
 *
 * Each digit animates independently with a slot machine effect.
 */
export const SlotNumber = memo(function SlotNumber({
  value,
  prefix = '',
  suffix = '',
  decimals = 2,
  className = '',
}: SlotNumberProps) {
  const [displayValue, setDisplayValue] = useState(value);

  useEffect(() => {
    setDisplayValue(value);
  }, [value]);

  // Format the number
  const formatted = displayValue.toFixed(decimals);
  const chars = formatted.split('');

  return (
    <span className={`inline-flex items-baseline ${className}`}>
      {prefix && <span>{prefix}</span>}

      {chars.map((char, index) => {
        // Non-numeric characters (decimal point, comma) render directly
        if (!/\d/.test(char)) {
          return (
            <span key={`sep-${index}`} className="mx-px">
              {char}
            </span>
          );
        }

        const digit = parseInt(char, 10);

        return (
          <span
            key={`digit-${index}`}
            className="slot-digit relative inline-block overflow-hidden text-center"
            style={{ height: '1em', width: '0.6em' }}
          >
            {digit}
          </span>
        );
      })}

      {suffix && <span>{suffix}</span>}
    </span>
  );
});
