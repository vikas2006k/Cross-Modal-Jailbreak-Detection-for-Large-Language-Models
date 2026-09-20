import React, { useState } from 'react';
import {
  Layers,
  Scale,
  RefreshCw,
  AlertTriangle,
  Brain,
  Sliders,
  RotateCcw,
} from 'lucide-react';
import { VerdictBadge } from '../components/common/VerdictBadge';
import { RiskGauge } from '../components/common/RiskGauge';
import { ImageDropzone } from '../components/common/ImageDropzone';
import { ErrorBoundary } from '../components/common/ErrorBoundary';
import { predictCrossModal, ApiError } from '../services/api';
import { saveScanRecord } from '../services/storage';
import type { CrossModalPredictResponse } from '../types/api';

export const CrossModalScanner: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CrossModalPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelected = (file: File, preview: string) => {
    setSelectedFile(file);
    setPreviewUrl(preview);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrompt('');
    setResult(null);
    setError(null);
  };

  const handleScan = async () => {
    if (!selectedFile) {
      setError('Please upload or select an image to initiate cross-modal fusion.');
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const res = await predictCrossModal(selectedFile, prompt.trim() || undefined, true);
      setResult(res);
      saveScanRecord({
        type: 'CROSS_MODAL',
        verdict: res.prediction,
        confidence: res.confidence ?? 0.95,
        riskScore: res.risk_score ?? 0,
        attackCategory: res.attack_category || 'Cross-Modal Assessment',
        latencyMs: res.latency_ms ?? res.inference_time_ms ?? 0,
        imageName: selectedFile.name,
        imageUrl: previewUrl || undefined,
        inputPrompt: prompt.trim() || undefined,
        ocrText: res.ocr_extracted_text || '',
        visionRisk: res.vision_risk_score ?? 0,
        textRisk: res.text_risk_score ?? 0,
        reason: res.fusion_reason,
      });
    } catch (err: unknown) {
      console.error('Cross-modal scan error:', err);
      if (err instanceof ApiError) {
        setError(`[HTTP ${err.statusCode || 'ERR'}] ${err.message}`);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Cross-modal fusion pipeline failed. Verify backend service connectivity.');
      }
    } finally {
      setLoading(false);
    }
  };

  const isJailbreak = result?.prediction === 'JAILBREAK';

  // Compute contribution percentages safely for visual breakdown
  const vWeight =
    typeof result?.modality_weights?.vision === 'number' && !isNaN(result.modality_weights.vision)
      ? result.modality_weights.vision
      : 0.35;
  const tWeight =
    typeof result?.modality_weights?.text === 'number' && !isNaN(result.modality_weights.text)
      ? result.modality_weights.text
      : 0.65;
  const totalW = vWeight + tWeight || 1;
  const visionPct = Math.round((vWeight / totalW) * 100);
  const textPct = 100 - visionPct;

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                Cross-Modal Fusion Security Engine
              </h1>
              <p className="text-xs text-slate-400">
                Core Security Engine: Adaptive Gating, False-Positive Suppression & Bimodal Consensus
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Multimodal Inputs */}
        <div className="lg:col-span-6 space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-5">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  Modality 1: Target Image
                </span>
                <span className="text-[11px] text-cyan-400 font-mono">Vision & OCR</span>
              </div>
              <ImageDropzone
                onImageSelected={handleImageSelected}
                selectedPreview={previewUrl}
                onClear={handleClear}
                disabled={loading}
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label
                  htmlFor="companionPrompt"
                  className="text-xs font-bold text-slate-300 uppercase tracking-wider"
                >
                  Modality 2: Companion User Prompt (Optional)
                </label>
                <span className="text-[11px] text-slate-500">Dual-Channel Sync</span>
              </div>
              <textarea
                id="companionPrompt"
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter any accompanying text prompt, query context, or leave blank to evaluate image standalone..."
                className="w-full rounded-xl bg-slate-950 border border-slate-800 p-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono resize-none"
              />
            </div>

            {error && (
              <div className="p-3.5 rounded-xl bg-red-950/60 border border-red-800 text-red-200 text-xs flex items-center justify-between gap-3 animate-in fade-in">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  <span>{error}</span>
                </div>
                <button
                  type="button"
                  onClick={handleScan}
                  className="px-2.5 py-1 rounded-lg bg-red-900/60 hover:bg-red-800 text-[11px] font-semibold text-white flex items-center gap-1 transition-colors shrink-0"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Retry</span>
                </button>
              </div>
            )}

            {/* Action Bar */}
            <div className="flex items-center justify-between pt-2">
              <button
                type="button"
                onClick={handleClear}
                disabled={loading}
                className="px-3 py-2 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
              >
                Reset Inputs
              </button>

              <button
                type="button"
                onClick={handleScan}
                disabled={loading || !selectedFile}
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-102"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Executing Fusion Gating Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Scale className="w-4 h-4" />
                    <span>Run Cross-Modal Fusion</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Fusion Decision & Modality Attribution with Error Boundary */}
        <div className="lg:col-span-6 space-y-6">
          <ErrorBoundary fallbackTitle="Cross-Modal Fusion Result Error" onReset={() => setResult(null)}>
            {result ? (
              <div className="space-y-4 animate-in fade-in duration-200">
                {/* Primary Decision Banner */}
                <div
                  className={`rounded-2xl p-6 border shadow-2xl space-y-5 transition-all ${
                    isJailbreak
                      ? 'bg-slate-900/90 border-red-600/40 shadow-red-950/40'
                      : 'bg-slate-900/90 border-emerald-600/40 shadow-emerald-950/40'
                  }`}
                >
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                      <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                        Unified Defense Decision
                      </span>
                      <div className="mt-1">
                        <VerdictBadge verdict={result.prediction} size="lg" />
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-slate-400 uppercase tracking-wider">
                        Confidence
                      </span>
                      <p className="text-2xl font-mono font-bold text-white mt-0.5">
                        {((result.confidence ?? 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Risk Gauge */}
                  <div className="py-2 bg-slate-950/60 rounded-xl border border-slate-800/80">
                    <RiskGauge
                      riskScore={result.risk_score ?? 0}
                      label="Fused Cross-Modal Risk Score"
                    />
                  </div>

                  {/* Mathematical Fusion Weight Contribution Bar */}
                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between text-xs text-slate-300 font-medium">
                      <span className="flex items-center gap-1.5">
                        <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                        Adaptive Fusion Contributions:
                      </span>
                      <span className="font-mono text-cyan-400">
                        Vision {visionPct}% | Text {textPct}%
                      </span>
                    </div>

                    {/* Dual-tone Progress bar */}
                    <div className="h-3 w-full rounded-full bg-slate-900 overflow-hidden flex border border-slate-800">
                      <div
                        className="h-full bg-gradient-to-r from-blue-600 to-cyan-500 transition-all duration-500"
                        style={{ width: `${visionPct}%` }}
                        title={`Vision Weight: ${visionPct}%`}
                      />
                      <div
                        className="h-full bg-gradient-to-r from-teal-500 to-emerald-500 transition-all duration-500"
                        style={{ width: `${textPct}%` }}
                        title={`Text Weight: ${textPct}%`}
                      />
                    </div>

                    <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1 font-mono">
                      <span>CLIP ViT-B/32 (w_v={vWeight.toFixed(2)})</span>
                      <span>DistilBERT + OCR (w_t={tWeight.toFixed(2)})</span>
                    </div>
                  </div>

                  {/* Reasoning Card */}
                  <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-2 text-xs">
                    <div className="flex items-center gap-1.5 font-bold text-slate-200">
                      <Brain className="w-4 h-4 text-cyan-400" />
                      <span>Decision Reasoning & Guardrail Logic:</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed font-mono text-[11px] bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
                      {result.fusion_reason || 'Cross-modal evaluation complete.'}
                    </p>
                  </div>
                </div>

                {/* Modalities Side-by-Side Breakdown Card */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-300">Vision Branch</span>
                      <VerdictBadge verdict={result.vision_prediction} size="sm" />
                    </div>
                    <div className="font-mono text-lg font-bold text-cyan-400">
                      {(result.vision_risk_score ?? 0).toFixed(1)}%
                    </div>
                    <p className="text-[10px] text-slate-400">CLIP visual embedding classifier</p>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-300">Text & OCR Branch</span>
                      <VerdictBadge verdict={result.text_prediction} size="sm" />
                    </div>
                    <div className="font-mono text-lg font-bold text-cyan-400">
                      {(result.text_risk_score ?? 0).toFixed(1)}%
                    </div>
                    <p className="text-[10px] text-slate-400">DistilBERT tokenized sequence</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full min-h-[440px] rounded-2xl border border-dashed border-slate-800 bg-slate-950/40 p-8 flex flex-col items-center justify-center text-center text-slate-500">
                <Layers className="w-12 h-12 text-slate-700 mb-3" />
                <p className="text-sm font-semibold text-slate-300">Awaiting Multimodal Inputs</p>
                <p className="text-xs text-slate-500 max-w-sm mt-1">
                  Select an image and companion prompt on the left to observe how Adaptive Gating
                  fuses visual and textual signals to eliminate false positives.
                </p>
              </div>
            )}
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
};
