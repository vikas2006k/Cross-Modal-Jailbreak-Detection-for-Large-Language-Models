import React, { useState, useEffect } from 'react';
import { X, ZoomIn, ZoomOut, RotateCcw, Download, Info } from 'lucide-react';

interface LightboxModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageUrl: string;
  title: string;
  category?: string;
  description?: string;
  metadata?: Record<string, string | number>;
}

export const LightboxModal: React.FC<LightboxModalProps> = ({
  isOpen,
  onClose,
  imageUrl,
  title,
  category,
  description,
  metadata,
}) => {
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Reset scale when image changes or modal opens
  useEffect(() => {
    setScale(1);
  }, [imageUrl, isOpen]);

  if (!isOpen) return null;

  const handleZoomIn = () => setScale((s) => Math.min(s + 0.25, 3.0));
  const handleZoomOut = () => setScale((s) => Math.max(s - 0.25, 0.5));
  const handleResetZoom = () => setScale(1);

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = `${title.toLowerCase().replace(/\s+/g, '_')}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-slate-950/90 backdrop-blur-xl animate-in fade-in duration-200">
      <div className="relative w-full max-w-5xl max-h-[92vh] flex flex-col rounded-2xl bg-slate-900/95 border border-cyan-500/40 shadow-2xl shadow-cyan-950/60 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-white tracking-tight">{title}</h3>
                {category && (
                  <span className="px-2 py-0.5 text-[11px] font-semibold rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                    {category}
                  </span>
                )}
              </div>
              {description && (
                <p className="text-xs text-slate-400 mt-0.5 line-clamp-1">{description}</p>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <div className="flex items-center bg-slate-800/80 rounded-lg p-1 border border-slate-700">
              <button
                type="button"
                onClick={handleZoomOut}
                disabled={scale <= 0.5}
                className="p-1.5 rounded hover:bg-slate-700 text-slate-300 disabled:opacity-30 transition-all"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs font-mono px-2 text-cyan-300 font-semibold min-w-[3.5rem] text-center">
                {Math.round(scale * 100)}%
              </span>
              <button
                type="button"
                onClick={handleZoomIn}
                disabled={scale >= 3.0}
                className="p-1.5 rounded hover:bg-slate-700 text-slate-300 disabled:opacity-30 transition-all"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={handleResetZoom}
                className="p-1.5 rounded hover:bg-slate-700 text-slate-300 ml-1 transition-all"
                title="Reset Zoom"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            <button
              type="button"
              onClick={handleDownload}
              className="p-2 rounded-lg bg-cyan-950/70 hover:bg-cyan-900 text-cyan-300 border border-cyan-800/60 transition-all"
              title="Download Figure"
            >
              <Download className="w-4 h-4" />
            </button>

            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-lg bg-slate-800 hover:bg-red-950 text-slate-300 hover:text-red-300 border border-slate-700 transition-all ml-2"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Viewport container */}
        <div className="relative flex-1 overflow-auto bg-slate-950 flex items-center justify-center p-6 min-h-[400px]">
          <div
            className="transition-transform duration-150 ease-out origin-center select-none"
            style={{ transform: `scale(${scale})` }}
          >
            <img
              src={imageUrl}
              alt={title}
              className="max-h-[68vh] max-w-full object-contain rounded-lg shadow-2xl border border-slate-800"
            />
          </div>
        </div>

        {/* Metadata Footer */}
        {metadata && (
          <div className="px-6 py-2.5 bg-slate-950/90 border-t border-slate-800/80 flex flex-wrap items-center gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
              <Info className="w-3.5 h-3.5" />
              Figure Metadata:
            </span>
            {Object.entries(metadata).map(([key, val]) => (
              <span key={key} className="flex items-center gap-1">
                <span className="text-slate-500 capitalize">{key}:</span>
                <span className="font-mono text-slate-200">{val}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
