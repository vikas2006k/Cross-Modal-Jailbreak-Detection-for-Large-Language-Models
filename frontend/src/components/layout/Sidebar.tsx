import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FileText,
  Image as ImageIcon,
  Layers,
  Sparkles,
  BarChart3,
  History,
  ShieldCheck,
} from 'lucide-react';

const NAV_ITEMS = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Text Scanner', path: '/text-scanner', icon: FileText },
  { name: 'Image Scanner', path: '/image-scanner', icon: ImageIcon },
  { name: 'Cross-Modal Scanner', path: '/cross-modal-scanner', icon: Layers, badge: 'Core Engine' },
  { name: 'Explainability', path: '/explainability', icon: Sparkles },
  { name: 'Analytics', path: '/analytics', icon: BarChart3 },
  { name: 'Scan Reports', path: '/reports', icon: History },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 shrink-0 hidden lg:block border-r border-slate-800/80 bg-slate-950/60 p-4 min-h-[calc(100vh-61px)]">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
          Core Modules
        </div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-all group ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-950/80 to-blue-950/80 text-cyan-300 border border-cyan-500/30 shadow-md shadow-cyan-950/40'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <Icon className="w-4 h-4 transition-transform group-hover:scale-110 text-cyan-400/80" />
                <span>{item.name}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded-md bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-300 border border-amber-500/30">
                  {item.badge}
                </span>
              )}
            </NavLink>
          );
        })}
      </div>

      {/* Model Spec Box */}
      <div className="mt-8 p-3.5 rounded-xl bg-slate-900/70 border border-slate-800 text-xs text-slate-400 space-y-2">
        <div className="flex items-center gap-1.5 font-semibold text-slate-200">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Active Guardrails</span>
        </div>
        <div className="space-y-1 text-[11px]">
          <div className="flex justify-between">
            <span className="text-slate-500">Text:</span>
            <span className="text-slate-300">DistilBERT (99.2% Acc)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Vision:</span>
            <span className="text-slate-300">CLIP ViT-B/32</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">OCR:</span>
            <span className="text-slate-300">EasyOCR Engine</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Fusion:</span>
            <span className="text-cyan-300 font-semibold">Adaptive Gated (98.3%)</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
