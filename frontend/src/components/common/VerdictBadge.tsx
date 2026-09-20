import React from 'react';
import { ShieldCheck, ShieldAlert, HelpCircle } from 'lucide-react';
import type { Verdict } from '../../types/api';

interface VerdictBadgeProps {
  verdict?: Verdict | string | null;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export const VerdictBadge: React.FC<VerdictBadgeProps> = ({
  verdict,
  size = 'md',
  showIcon = true,
}) => {
  const normalized = (verdict || 'SAFE').toUpperCase();
  const isSafe = normalized === 'SAFE';
  const isJailbreak = normalized === 'JAILBREAK';

  const sizeClasses = {
    sm: 'text-xs px-2.5 py-0.5 gap-1',
    md: 'text-sm px-3.5 py-1 gap-1.5',
    lg: 'text-base px-5 py-2 gap-2 font-bold',
  };

  const iconSizes = {
    sm: 'w-3.5 h-3.5',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  };

  const colorClasses = isSafe
    ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-500/40 shadow-emerald-950/50'
    : isJailbreak
    ? 'bg-red-950/80 text-red-400 border border-red-500/50 shadow-red-950/60 animate-pulse'
    : 'bg-slate-900 text-slate-300 border border-slate-700';

  return (
    <span
      className={`inline-flex items-center rounded-full font-bold uppercase tracking-wider transition-all shadow-sm ${
        sizeClasses[size]
      } ${colorClasses}`}
    >
      {showIcon &&
        (isSafe ? (
          <ShieldCheck className={`${iconSizes[size]} text-emerald-400`} />
        ) : isJailbreak ? (
          <ShieldAlert className={`${iconSizes[size]} text-red-400`} />
        ) : (
          <HelpCircle className={`${iconSizes[size]} text-slate-400`} />
        ))}
      <span>{normalized}</span>
    </span>
  );
};
