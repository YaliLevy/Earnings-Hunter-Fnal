import { useEffect, useState } from 'react';

interface ConfidenceRingProps {
  confidence: number; // 0-1
  prediction: 'Growth' | 'Risk' | 'Stagnation';
}

export function ConfidenceRing({ confidence, prediction }: ConfidenceRingProps) {
  const [animatedOffset, setAnimatedOffset] = useState(440); // Start at 0%

  // SVG circle parameters
  const size = 180;
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius; // ~440 for r=70

  // Calculate stroke-dashoffset (0 = full, circumference = empty)
  const targetOffset = circumference * (1 - confidence);

  // Animate on mount/change
  useEffect(() => {
    // Small delay to ensure CSS transition works
    const timeout = setTimeout(() => {
      setAnimatedOffset(targetOffset);
    }, 50);
    return () => clearTimeout(timeout);
  }, [targetOffset]);

  // Colors based on prediction
  const colors = {
    Growth: { ring: '#00F090', text: 'BULLISH' },
    Risk: { ring: '#FF2E50', text: 'BEARISH' },
    Stagnation: { ring: '#f59e0b', text: 'NEUTRAL' },
  };

  const { ring: ringColor, text: verdictText } = colors[prediction];

  return (
    <div className="flex flex-col items-center py-4">
      {/* SVG Ring */}
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="transform -rotate-90"
        >
          {/* Background track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="hsl(220, 15%, 20%)"
            strokeWidth={strokeWidth}
          />

          {/* Progress ring */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={ringColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={animatedOffset}
            style={{
              transition: 'stroke-dashoffset 1s ease-out',
            }}
          />
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-5xl font-mono font-bold"
            style={{ color: ringColor }}
          >
            {Math.round(confidence * 100)}%
          </span>
          <span className="text-xs tracking-widest text-text-muted mt-1">
            CONFIDENCE
          </span>
        </div>
      </div>

      {/* Verdict */}
      <div
        className="text-xl font-bold tracking-wider mt-4"
        style={{ color: ringColor }}
      >
        {verdictText}
      </div>
    </div>
  );
}
