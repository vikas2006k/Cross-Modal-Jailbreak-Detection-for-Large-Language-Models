# Table II: Subsystem Hyperparameter & Training Configuration

Comprehensive specification of training hyperparameters, optimizers, and architectural parameters across the Text and Vision subsystems.

| Subsystem / Component | Hyperparameter / Setting | Value | Rationale / Reference |
| :--- | :--- | :--- | :--- |
| **DistilBERT Text Detector** | Pretrained Backbone | `distilbert-base-uncased` | 66M parameters, 6 transformer layers, 768 hidden size |
| | Classification Head | Linear $(768 \rightarrow 2)$ + Dropout $(p=0.2)$ | Binary sequence classification (SAFE vs JAILBREAK) |
| | Max Sequence Length | 128 tokens | Covers 98.4% of CMJD-10K prompt distributions |
| | Optimizer | AdamW | $\beta_1=0.9, \beta_2=0.999, \epsilon=1\times 10^{-8}$ |
| | Initial Learning Rate | $2.0 \times 10^{-5}$ | Linear warmup over first 10% steps, cosine decay |
| | Weight Decay | $0.01$ | $L_2$ regularization on non-bias parameters |
| | Batch Size | 16 | Optimal convergence on standard hardware |
| | Training Epochs | 4 (Early Stopped at Ep. 2) | Validation $F_1$ monitored with patience = 2 |
| | Loss Function | Binary Cross-Entropy with Logits | $\mathcal{L}_{\text{BCE}} = - [y \log \hat{y} + (1-y) \log (1-\hat{y})]$ |
| **CLIP Vision Detector** | Visual Backbone | OpenAI CLIP ViT-B/32 (Frozen) | 12 transformer layers, 12 attention heads, 512-dim embedding |
| | Classification Head | MLP: $512 \rightarrow 256 \rightarrow 64 \rightarrow 2$ | Non-linear projection with ReLU and BatchNorm |
| | Dropout Rate | $p = 0.30$ | Prevents overfitting on photographic backgrounds |
| | Optimizer | Adam | Learning rate $\eta = 1.0 \times 10^{-3}$ |
| | Image Input Resolution | $224 \times 224 \times 3$ pixels | Standard CLIP ViT patch input $(32 \times 32$ patch size) |
| | Batch Size | 32 | Mini-batch stochastic optimization |
| | Training Epochs | 10 | Converged at epoch 8 with zero loss plateau |
| **EasyOCR Engine** | OCR Model | CRAFT + CRNN (PyTorch) | Character Region Awareness for Text Detection |
| | Target Languages | English (`['en']`) | Optimized for alphanumeric adversarial directives |
| | Detection Threshold | 0.40 | High-recall character boundary localization |
| **Adaptive Fusion Engine** | Default Weights | $w_{\text{vision}} = 0.35, w_{\text{text}} = 0.65$ | Calibrated to text primacy in prompt injections |
| | Benign Gating Trigger | OCR Length $> 120$ chars, zero regex | Active semantic discount on academic typography |
| | Entropy Suppression | Shannon $H(X) < 2.0$ bits | Eliminates spurious visual false positives on textless canvases |
