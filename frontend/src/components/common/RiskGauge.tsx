import React from 'react';

interface RiskGaugeProps {
  score?: number; // 0 - 100
  riskScore?: number;
  size?: number;
  label?: string;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({
  score,
  riskScore,
  size = 180,
  label = 'Risk Index',
}) => {
  const rawValue = typeof riskScore === 'number' ? riskScore : (typeof score === 'number' ? score : 0);
  const validValue = Number.isFinite(rawValue) ? rawValue : 0;
  const normalizedScore = Math.max(0, Math.min(100, validValue));

  // Determine color and status
  let color = '#10b981'; // Green
  let statusText = 'Low Threat';
  if (normalizedScore > 75) {
    color = '#ef4444'; // Red
    statusText = 'Critical Jailbreak';
  } else if (normalizedScore >= 50) {
    color = '#f97316'; // Orange
    statusText = 'Elevated Risk';
  } else if (normalizedScore >= 25) {
    color = '#eab308'; // Yellow
    statusText = 'Moderate Risk';
  }

  // Semi-circle SVG calculation
  const strokeWidth = 14;
  const radius = Math.max(10, (size - strokeWidth * 2) / 2);
  const circumference = Math.PI * radius; // Half circle
  const strokeDashoffset = circumference - (normalizedScore / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center p-2">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size / 1.7 }}>
        <svg
          width={size}
          height={size / 1.5}
          viewBox={`0 0 ${size} ${size / 1.5}`}
          className="overflow-visible"
        >
          {/* Background Track */}
          <path
            d={`M ${strokeWidth} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth} ${size / 2}`}
            fill="none"
            stroke="#1e293b"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Animated Active Track */}
          <path
            d={`M ${strokeWidth} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth} ${size / 2}`}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{
              transition: 'stroke-dashoffset 0.8s ease-in-out, stroke 0.4s ease',
              filter: `drop-shadow(0 0 8px ${color}80)`,
            }}
          />
        </svg>

        {/* Center Score Readout */}
        <div className="absolute bottom-2 flex flex-col items-center">
          <span
            className="text-3xl font-extrabold tracking-tight font-mono"
            style={{ color, textShadow: `0 0 14px ${color}60` }}
          >
            {normalizedScore.toFixed(1)}%
          </span>
          <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
            {statusText}
          </span>
        </div>
      </div>
      <span className="text-xs text-slate-400 mt-1 font-medium">{label}</span>
    </div>
  );
};
