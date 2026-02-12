import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from 'recharts';
import type { GoldenTriangle } from '../../types';

interface GoldenTriangleRadarProps {
  data: GoldenTriangle;
}

export function GoldenTriangleRadar({ data }: GoldenTriangleRadarProps) {
  // Transform data for Recharts
  const chartData = [
    {
      axis: 'Financial',
      value: data.financial.score * 10, // Convert 0-10 to 0-100
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

  return (
    <div className="w-full">
      {/* Title */}
      <h3 className="text-xs tracking-widest text-text-muted mb-4 px-2">
        GOLDEN TRIANGLE
      </h3>

      {/* Radar Chart */}
      <ResponsiveContainer width="100%" height={200}>
        <RadarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
          <PolarGrid
            stroke="hsl(220, 15%, 20%)"
            strokeDasharray="3 3"
          />
          <PolarAngleAxis
            dataKey="axis"
            tick={{
              fill: 'hsl(220, 10%, 50%)',
              fontSize: 10,
              fontWeight: 500,
            }}
          />
          <Radar
            name="Score"
            dataKey="value"
            stroke="#00F090"
            fill="#00F090"
            fillOpacity={0.25}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>

      {/* Legend with weights */}
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
    </div>
  );
}
