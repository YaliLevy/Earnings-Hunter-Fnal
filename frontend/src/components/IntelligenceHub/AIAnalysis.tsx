interface AIAnalysisProps {
  symbol: string;
  companyName: string | null;
  prediction: 'Growth' | 'Risk' | 'Stagnation';
  confidence: number;
  epsBeat: boolean | null;
  epsSurprise: number | null;
  ceoPhrases: string[];
  deepInsight?: string | null;
}

export function AIAnalysis({
  symbol,
  companyName,
  prediction,
  confidence,
  epsBeat,
  epsSurprise,
  ceoPhrases,
  deepInsight,
}: AIAnalysisProps) {
  const company = companyName || symbol;

  // Generate analysis text based on data
  const generateAnalysis = () => {
    const sentiment =
      prediction === 'Growth'
        ? 'exceptional'
        : prediction === 'Risk'
        ? 'concerning'
        : 'mixed';

    const outlook =
      prediction === 'Growth'
        ? 'strong upside potential'
        : prediction === 'Risk'
        ? 'elevated risk'
        : 'sideways movement';

    let text = `${company} demonstrates ${sentiment} fundamental strength `;

    if (epsBeat !== null && epsSurprise !== null) {
      if (epsBeat) {
        text += `with a significant ${Math.abs(epsSurprise).toFixed(1)}% EPS beat, `;
      } else {
        text += `despite a ${Math.abs(epsSurprise).toFixed(1)}% EPS miss, `;
      }
    }

    text += `indicating ${outlook}. `;

    if (ceoPhrases.length > 0) {
      text += `CEO tone highlights: "${ceoPhrases[0]}"`;
    }

    return text;
  };

  return (
    <div className="mt-6">
      {/* Title */}
      <h3 className="text-xs tracking-widest text-purple mb-3">
        AI ANALYSIS
      </h3>

      {/* Generated Analysis */}
      <div className="bg-surface/50 border border-border rounded-lg p-4">
        <p className="text-sm text-text-secondary leading-relaxed">
          {deepInsight || generateAnalysis()}
        </p>
      </div>

      {/* Confidence indicator */}
      <div className="mt-3 flex items-center gap-2">
        <div className="flex-1 h-1 bg-border rounded-full overflow-hidden">
          <div
            className="h-full bg-purple rounded-full transition-all duration-500"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
        <span className="text-xs text-text-muted font-mono">
          {(confidence * 100).toFixed(0)}% conf
        </span>
      </div>
    </div>
  );
}
