import { useEffect, useState } from 'react';
import type { CompositeBreakdown } from '../../types';

interface ConfidenceRingProps {
  confidence: number; // 0-1
  prediction: 'Growth' | 'Risk' | 'Stagnation';
  breakdown?: CompositeBreakdown | null;
}

function getBarColor(score: number): string {
  if (score >= 65) return '#10b981'; // green
  if (score >= 35) return '#f59e0b'; // amber
  return '#ef4444'; // red
}

export function ConfidenceRing({ confidence, prediction, breakdown }: ConfidenceRingProps) {
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

      {/* Score Breakdown */}
      {breakdown && breakdown.components && (
        <div className="w-full mt-5 bg-surface border border-border rounded-lg p-4">
          <h4 className="text-xs tracking-widest text-text-muted mb-4">
            SCORE BREAKDOWN
          </h4>

          <div className="space-y-3">
            {breakdown.components.map((comp) => (
              <div key={comp.key} className="group">
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-text-muted">{comp.label.toUpperCase()}</span>
                  <span
                    className="text-lg font-mono font-semibold"
                    style={{ color: getBarColor(comp.score) }}
                  >
                    {Math.round(comp.score)}
                  </span>
                </div>
                {/* Progress bar */}
                <div className="h-1 bg-border rounded-full overflow-hidden mt-1">
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{
                      width: `${Math.min(comp.score, 100)}%`,
                      backgroundColor: getBarColor(comp.score),
                    }}
                  />
                </div>
                {/* Reasoning on hover */}
                {comp.reasoning && (
                  <div className="hidden group-hover:block mt-1 text-[10px] text-text-muted/70 leading-tight">
                    {comp.reasoning}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Overall reasoning */}
          {breakdown.overall_reasoning && (
            <>
              <div className="h-px bg-border my-4" />
              <p className="text-[10px] text-text-muted leading-relaxed">
                {breakdown.overall_reasoning}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
