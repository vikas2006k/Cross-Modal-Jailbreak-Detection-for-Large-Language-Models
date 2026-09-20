import React, { useState } from 'react';
import {
  Sparkles,
  Eye,
  ZoomIn,
} from 'lucide-react';
import { LightboxModal } from '../components/common/LightboxModal';

interface FigureItem {
  id: string;
  title: string;
  category: 'shap_lime' | 'attention' | 'vision' | 'ocr' | 'fusion';
  categoryLabel: string;
  description: string;
  imageSrc: string;
  resolution: string;
  badge: string;
}

const FIGURES: FigureItem[] = [
  {
    id: 'combined-syn',
    title: 'End-to-End Multimodal Explainability Synthesis',
    category: 'fusion',
    categoryLabel: 'Fusion & Benchmark',
    description:
      'Tri-panel synthesis uniting OCR spatial bounding boxes, DistilBERT token attention, and CLIP visual saliency.',
    imageSrc: '/figures/combined_explanation_sample.png',
    resolution: '300 DPI Publication Quality',
    badge: 'Core Architecture Figure',
  },
  {
    id: 'shap-global',
    title: 'DistilBERT SHAP Global Token Importance',
    category: 'shap_lime',
    categoryLabel: 'SHAP & LIME',
    description:
      'Mean absolute SHAP value impact across CMJD-10K vocabulary highlighting top malicious trigger terms.',
    imageSrc: '/figures/text_shap_global_importance.png',
    resolution: 'Vector / 300 DPI',
    badge: 'SHAP Summary',
  },
  {
    id: 'shap-local',
    title: 'SHAP Local Feature Force Explanations',
    category: 'shap_lime',
    categoryLabel: 'SHAP & LIME',
    description:
      'Local token-level attribution forces pushing prediction toward JAILBREAK vs SAFE on held-out test prompts.',
    imageSrc: '/figures/text_shap_local_summary.png',
    resolution: '300 DPI',
    badge: 'SHAP Local',
  },
  {
    id: 'lime-summary',
    title: 'LIME Local Interpretable Surrogate Model',
    category: 'shap_lime',
    categoryLabel: 'SHAP & LIME',
    description:
      'Linear surrogate boundary approximations proving robustness around adversarial prompt injection phrases.',
    imageSrc: '/figures/text_lime_summary.png',
    resolution: '300 DPI',
    badge: 'LIME Explanations',
  },
  {
    id: 'attn-multi',
    title: 'DistilBERT Multi-Head Attention Heatmap',
    category: 'attention',
    categoryLabel: 'DistilBERT Attention',
    description:
      'Self-attention weights across transformer heads capturing long-range dependency on override verbs.',
    imageSrc: '/figures/text_attention_multiclass_heatmap.png',
    resolution: '300 DPI',
    badge: 'Self-Attention',
  },
  {
    id: 'vision-saliency-jb',
    title: 'CLIP ViT-B/32 Adversarial Saliency Map',
    category: 'vision',
    categoryLabel: 'Vision & Saliency',
    description:
      'Grad-CAM gradient activations localized tightly around typographic injection headers in poster images.',
    imageSrc: '/figures/vision_saliency_sample_1_jailbreak.png',
    resolution: '224x224 Patch Grid',
    badge: 'CLIP Grad-CAM',
  },
  {
    id: 'vision-saliency-safe',
    title: 'CLIP ViT-B/32 Benign Saliency Map',
    category: 'vision',
    categoryLabel: 'Vision & Saliency',
    description:
      'Evenly distributed attention activations demonstrating absence of adversarial typographic focal points.',
    imageSrc: '/figures/vision_saliency_sample_3_safe.png',
    resolution: '224x224 Patch Grid',
    badge: 'CLIP Grad-CAM',
  },
  {
    id: 'ocr-safe',
    title: 'EasyOCR Spatial Bounding Box (Educational Slide)',
    category: 'ocr',
    categoryLabel: 'EasyOCR Bounding Boxes',
    description:
      'Precise spatial localization of multi-line typography on academic slides without false malicious alerts.',
    imageSrc: '/figures/vision_ocr_vis_1_safe.png',
    resolution: 'High Res OCR Map',
    badge: 'EasyOCR Vis',
  },
  {
    id: 'ocr-jb',
    title: 'EasyOCR Spatial Bounding Box (Jailbreak Poster)',
    category: 'ocr',
    categoryLabel: 'EasyOCR Bounding Boxes',
    description:
      'Extraction of camouflaged prompt injection directives embedded inside stylized visual banners.',
    imageSrc: '/figures/vision_ocr_vis_2_jailbreak.png',
    resolution: 'High Res OCR Map',
    badge: 'EasyOCR Vis',
  },
  {
    id: 'fp-reduction',
    title: 'False-Positive Reduction Impact',
    category: 'fusion',
    categoryLabel: 'Fusion & Benchmark',
    description:
      'Quantitative demonstration of false-positive reduction from 16.0% down to 3.3% using adaptive gating.',
    imageSrc: '/figures/false_positive_reduction_chart.png',
    resolution: '300 DPI',
    badge: 'Ablation Chart',
  },
  {
    id: 'modality-dist',
    title: 'Adaptive Modality Contribution Distribution',
    category: 'fusion',
    categoryLabel: 'Fusion & Benchmark',
    description:
      'Dynamic shift of fusion weights depending on OCR presence and visual entropy across N=120 samples.',
    imageSrc: '/figures/modality_contribution_distribution.png',
    resolution: '300 DPI',
    badge: 'Weight Distribution',
  },
  {
    id: 'confusion-matrix',
    title: 'Validation Benchmark Confusion Matrix (120 Samples)',
    category: 'fusion',
    categoryLabel: 'Fusion & Benchmark',
    description:
      '118/120 accurate classifications with 100% jailbreak recall and 96.77% precision on challenging test cases.',
    imageSrc: '/figures/confusion_matrix.png',
    resolution: '300 DPI',
    badge: 'Confusion Matrix',
  },
  {
    id: 'roc-curve',
    title: 'Receiver Operating Characteristic (ROC-AUC = 0.9854)',
    category: 'fusion',
    categoryLabel: 'Fusion & Benchmark',
    description:
      'Cross-modal fusion ROC curve demonstrating superior discrimination compared to single-modality baselines.',
    imageSrc: '/figures/roc_curve.png',
    resolution: '300 DPI',
    badge: 'ROC-AUC Curve',
  },
];

const CATEGORIES = [
  { key: 'all', label: 'All Figures' },
  { key: 'fusion', label: 'Fusion & Benchmark' },
  { key: 'shap_lime', label: 'SHAP & LIME' },
  { key: 'attention', label: 'DistilBERT Attention' },
  { key: 'vision', label: 'Vision & Saliency' },
  { key: 'ocr', label: 'OCR Bounding Boxes' },
];

export const Explainability: React.FC = () => {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [selectedFigure, setSelectedFigure] = useState<FigureItem | null>(null);

  const filteredFigures =
    activeCategory === 'all'
      ? FIGURES
      : FIGURES.filter((f) => f.category === activeCategory);

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                Explainability & Interpretability Gallery
              </h1>
              <p className="text-xs text-slate-400">
                Explainability & Diagnostic Evidence: SHAP, LIME, Attention Heatmaps, OCR Spatial Maps, and
                CLIP Saliency
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Category Filter Pills */}
      <div className="flex flex-wrap items-center gap-2">
        {CATEGORIES.map((cat) => (
          <button
            key={cat.key}
            type="button"
            onClick={() => setActiveCategory(cat.key)}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              activeCategory === cat.key
                ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-md shadow-cyan-950/50'
                : 'bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800'
            }`}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Figure Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredFigures.map((fig) => (
          <div
            key={fig.id}
            className="group rounded-2xl overflow-hidden bg-slate-900/80 border border-slate-800 hover:border-cyan-500/50 transition-all duration-300 shadow-xl hover:-translate-y-1 flex flex-col"
          >
            {/* Figure Image Preview Container */}
            <div className="relative h-56 bg-slate-950 overflow-hidden flex items-center justify-center p-3 border-b border-slate-800/80">
              <img
                src={fig.imageSrc}
                alt={fig.title}
                className="max-h-full max-w-full object-contain rounded transition-transform duration-300 group-hover:scale-105"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3 backdrop-blur-xs">
                <button
                  type="button"
                  onClick={() => setSelectedFigure(fig)}
                  className="p-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold shadow-lg transition-transform hover:scale-110 flex items-center gap-1.5 text-xs"
                >
                  <ZoomIn className="w-4 h-4" />
                  <span>Inspect & Zoom</span>
                </button>
              </div>

              {/* Top Badge */}
              <div className="absolute top-2.5 left-2.5">
                <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-slate-900/90 text-cyan-300 border border-cyan-800/50 backdrop-blur-sm">
                  {fig.badge}
                </span>
              </div>
            </div>

            {/* Content Details */}
            <div className="p-5 flex-1 flex flex-col justify-between space-y-3">
              <div>
                <div className="text-[11px] font-semibold text-cyan-400 uppercase tracking-wider mb-1">
                  {fig.categoryLabel}
                </div>
                <h3 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors line-clamp-1">
                  {fig.title}
                </h3>
                <p className="text-xs text-slate-400 mt-1.5 line-clamp-2 leading-relaxed">
                  {fig.description}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500">
                <span>{fig.resolution}</span>
                <button
                  type="button"
                  onClick={() => setSelectedFigure(fig)}
                  className="text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1 transition-colors"
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>View Details</span>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Interactive Lightbox Viewer */}
      {selectedFigure && (
        <LightboxModal
          isOpen={!!selectedFigure}
          onClose={() => setSelectedFigure(null)}
          imageUrl={selectedFigure.imageSrc}
          title={selectedFigure.title}
          category={selectedFigure.categoryLabel}
          description={selectedFigure.description}
          metadata={{
            category: selectedFigure.categoryLabel,
            resolution: selectedFigure.resolution,
            badge: selectedFigure.badge,
            license: 'Enterprise Model Artifact',
          }}
        />
      )}
    </div>
  );
};
