import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-slate-800/80 bg-slate-950/90 py-4 px-6 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
      <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
        <span className="font-bold text-slate-300">SentinelGuard AI v1.0</span>
        <span className="hidden sm:inline text-slate-600">—</span>
        <span className="text-slate-400">Cross-Modal AI Security Platform</span>
      </div>
      <div className="text-slate-400 text-[11px]">
        Built using DistilBERT, CLIP, EasyOCR, FastAPI, and React.
      </div>
    </footer>
  );
};
