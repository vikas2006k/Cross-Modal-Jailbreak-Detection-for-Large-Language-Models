import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  color?: 'cyan' | 'emerald' | 'crimson' | 'amber' | 'blue' | 'purple';
  trend?: {
    value: string;
    positive: boolean;
  };
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  color = 'cyan',
  trend,
  className = '',
}) => {
  const colorMap = {
    cyan: {
      bg: 'from-cyan-950/40 to-blue-950/20',
      border: 'border-cyan-800/40 hover:border-cyan-500/50',
      iconBg: 'bg-cyan-950/80 border-cyan-500/30 text-cyan-400',
      glow: 'shadow-cyan-950/30',
      textAccent: 'text-cyan-400',
    },
    emerald: {
      bg: 'from-emerald-950/40 to-teal-950/20',
      border: 'border-emerald-800/40 hover:border-emerald-500/50',
      iconBg: 'bg-emerald-950/80 border-emerald-500/30 text-emerald-400',
      glow: 'shadow-emerald-950/30',
      textAccent: 'text-emerald-400',
    },
    crimson: {
      bg: 'from-red-950/40 to-rose-950/20',
      border: 'border-red-800/40 hover:border-red-500/50',
      iconBg: 'bg-red-950/80 border-red-500/30 text-red-400',
      glow: 'shadow-red-950/30',
      textAccent: 'text-red-400',
    },
    amber: {
      bg: 'from-amber-950/40 to-yellow-950/20',
      border: 'border-amber-800/40 hover:border-amber-500/50',
      iconBg: 'bg-amber-950/80 border-amber-500/30 text-amber-400',
      glow: 'shadow-amber-950/30',
      textAccent: 'text-amber-400',
    },
    blue: {
      bg: 'from-blue-950/40 to-indigo-950/20',
      border: 'border-blue-800/40 hover:border-blue-500/50',
      iconBg: 'bg-blue-950/80 border-blue-500/30 text-blue-400',
      glow: 'shadow-blue-950/30',
      textAccent: 'text-blue-400',
    },
    purple: {
      bg: 'from-purple-950/40 to-indigo-950/20',
      border: 'border-purple-800/40 hover:border-purple-500/50',
      iconBg: 'bg-purple-950/80 border-purple-500/30 text-purple-400',
      glow: 'shadow-purple-950/30',
      textAccent: 'text-purple-400',
    },
  };

  const currentTheme = colorMap[color];

  return (
    <div
      className={`relative overflow-hidden rounded-xl p-5 bg-gradient-to-br ${currentTheme.bg} border ${currentTheme.border} backdrop-blur-md shadow-lg ${currentTheme.glow} transition-all duration-300 hover:-translate-y-0.5 ${className}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400 mb-1">
            {title}
          </p>
          <h3 className="text-2xl font-bold tracking-tight text-white font-mono">
            {value}
          </h3>
        </div>
        <div className={`p-2.5 rounded-xl border ${currentTheme.iconBg} shadow-inner`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between text-xs">
        {subtitle && <span className="text-slate-400">{subtitle}</span>}
        {trend && (
          <span
            className={`font-semibold ml-auto ${
              trend.positive ? 'text-emerald-400' : 'text-red-400'
            }`}
          >
            {trend.value}
          </span>
        )}
      </div>
    </div>
  );
};
