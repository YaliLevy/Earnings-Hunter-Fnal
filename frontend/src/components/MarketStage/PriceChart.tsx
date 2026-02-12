import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { HistoricalData } from '../../types';

interface PriceChartProps {
  data: HistoricalData | null;
  symbol: string;
}

export function PriceChart({ data, symbol }: PriceChartProps) {
  if (!data || data.prices.length === 0) {
    return <PriceChartEmpty />;
  }

  // Combine price data with SMA and forecast for charting
  const chartData = data.prices.map((p, i) => ({
    date: formatDate(p.date),
    rawDate: p.date,
    price: p.close,
    sma20: data.sma20[i],
    isHistorical: true,
  }));

  // Add forecast data points
  const forecastData = data.forecast.map((p) => ({
    date: formatDate(p.date),
    rawDate: p.date,
    forecast: p.close,
    isHistorical: false,
  }));

  const allData = [...chartData, ...forecastData];

  // Determine if positive or negative trend
  const firstPrice = data.prices[0]?.close || 0;
  const lastPrice = data.prices[data.prices.length - 1]?.close || 0;
  const isPositive = lastPrice >= firstPrice;

  const gradientColor = isPositive ? '#00F090' : '#FF2E50';
  const lineColor = isPositive ? '#00F090' : '#FF2E50';

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart
        data={allData}
        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={gradientColor} stopOpacity={0.3} />
            <stop offset="100%" stopColor={gradientColor} stopOpacity={0} />
          </linearGradient>
        </defs>

        <CartesianGrid
          strokeDasharray="3 3"
          stroke="hsl(220, 15%, 20%)"
          strokeOpacity={0.5}
          vertical={false}
        />

        <XAxis
          dataKey="date"
          stroke="hsl(220, 10%, 50%)"
          fontSize={10}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />

        <YAxis
          stroke="hsl(220, 10%, 50%)"
          fontSize={10}
          tickLine={false}
          axisLine={false}
          orientation="right"
          domain={['auto', 'auto']}
          tickFormatter={(value) => `$${value.toFixed(0)}`}
        />

        <Tooltip
          contentStyle={{
            background: 'hsl(225, 15%, 7%)',
            border: '1px solid hsl(220, 15%, 20%)',
            borderRadius: '8px',
            fontSize: '12px',
          }}
          labelStyle={{ color: 'hsl(220, 10%, 50%)' }}
          formatter={(value: number, name: string) => {
            if (name === 'price') return [`$${value.toFixed(2)}`, 'Price'];
            if (name === 'sma20') return [`$${value.toFixed(2)}`, 'SMA(20)'];
            if (name === 'forecast') return [`$${value.toFixed(2)}`, 'Forecast'];
            return [value, name];
          }}
        />

        {/* Main price area */}
        <Area
          type="monotone"
          dataKey="price"
          stroke={lineColor}
          strokeWidth={2}
          fill="url(#priceGradient)"
          dot={false}
          activeDot={{ r: 4, fill: lineColor }}
        />

        {/* SMA(20) line */}
        <Line
          type="monotone"
          dataKey="sma20"
          stroke="hsl(220, 10%, 50%)"
          strokeWidth={1.5}
          dot={false}
          connectNulls={false}
        />

        {/* Forecast line (dotted) */}
        <Line
          type="monotone"
          dataKey="forecast"
          stroke="hsl(220, 10%, 50%)"
          strokeWidth={1.5}
          strokeDasharray="6 4"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function PriceChartEmpty() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-text-muted">
      <div className="text-lg mb-2">Search a ticker to load chart</div>
      <div className="text-sm text-text-muted/60">Press ⌘K to open search</div>
    </div>
  );
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}
