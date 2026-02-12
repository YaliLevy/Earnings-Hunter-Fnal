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
            className="slot-digit relative inline-block overflow-hidden"
            style={{ height: '1em' }}
          >
            <span
              className="slot-digit-inner absolute inset-x-0"
              style={{
                transform: `translateY(-${digit * 10}%)`,
                transition: 'transform 0.6s cubic-bezier(0.23, 1, 0.32, 1)',
              }}
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
                <span
                  key={n}
                  className="block"
                  style={{ height: '1em', lineHeight: '1em' }}
                >
                  {n}
                </span>
              ))}
            </span>
          </span>
        );
      })}

      {suffix && <span>{suffix}</span>}
    </span>
  );
});
