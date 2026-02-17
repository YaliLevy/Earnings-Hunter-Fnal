import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';
import type { GoldenTriangle, CompositeBreakdown } from '../../types';

interface GoldenTriangleRadarProps {
  data: GoldenTriangle;
  breakdown?: CompositeBreakdown | null;
}

function getScoreColor(score: number): string {
  if (score >= 65) return '#10b981';
  if (score >= 35) return '#f59e0b';
  return '#ef4444';
}

export function GoldenTriangleRadar({ data, breakdown }: GoldenTriangleRadarProps) {
  // Use composite breakdown (5 components) if available, otherwise fall back to golden triangle (3)
  const chartData = breakdown?.components
    ? breakdown.components.map((comp) => ({
        axis: comp.label,
        value: comp.score,
        fullMark: 100,
        weight: `${comp.weight}%`,
      }))
    : [
        {
          axis: 'Financial',
          value: data.financial.score * 10,
          fullMark: 100,
          weight: `${(data.financial.weight * 100).toFixed(0)}%`,
        },
        {
          axis: 'CEO Tone',
          value: data.ceo_tone.score * 10,
          fullMark: 100,
          weight: `${(data.ceo_tone.weight * 100).toFixed(0)}%`,
        },
        {
          axis: 'Sentiment',
          value: data.social.score * 10,
          fullMark: 100,
          weight: `${(data.social.weight * 100).toFixed(0)}%`,
        },
      ];

  const hasBreakdown = !!breakdown?.components;
  const components = breakdown?.components;

  return (
    <div className="w-full">
      {/* Title */}
      <h3 className="text-xs tracking-widest text-text-muted mb-4 px-2">
        {hasBreakdown ? 'ANALYSIS RADAR' : 'GOLDEN TRIANGLE'}
      </h3>

      {/* Radar Chart */}
      <ResponsiveContainer width="100%" height={hasBreakdown ? 220 : 200}>
        <RadarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
          <PolarGrid
            stroke="hsl(220, 15%, 20%)"
            strokeDasharray="3 3"
          />
          <PolarAngleAxis
            dataKey="axis"
            tick={{
              fill: 'hsl(220, 10%, 50%)',
              fontSize: 9,
              fontWeight: 500,
            }}
          />
          <Radar
            name="Score"
            dataKey="value"
            stroke="#00F090"
            fill="#00F090"
            fillOpacity={0.2}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>

      {/* Legend */}
      {hasBreakdown && components ? (
        <div className="grid grid-cols-3 gap-x-2 gap-y-3 mt-3 px-2">
          {components.map((comp) => (
            <div key={comp.key} className="text-center">
              <div
                className="text-base font-mono font-semibold"
                style={{ color: getScoreColor(comp.score) }}
              >
                {Math.round(comp.score)}
              </div>
              <div className="text-[8px] text-text-muted leading-tight">
                {comp.label.toUpperCase()}
                <br />
                ({comp.weight}%)
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-2 mt-4 px-2">
          <div className="text-center">
            <div className="text-lg font-mono font-semibold text-signal-green">
              {data.financial.score.toFixed(1)}
            </div>
            <div className="text-[9px] text-text-muted">
              FINANCIAL (40%)
            </div>
          </div>
          <div className="text-center">
            <div className="text-lg font-mono font-semibold text-signal-green">
              {data.ceo_tone.score.toFixed(1)}
            </div>
            <div className="text-[9px] text-text-muted">
              CEO TONE (35%)
            </div>
          </div>
          <div className="text-center">
            <div className="text-lg font-mono font-semibold text-signal-green">
              {data.social.score.toFixed(1)}
            </div>
            <div className="text-[9px] text-text-muted">
              SENTIMENT (25%)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
