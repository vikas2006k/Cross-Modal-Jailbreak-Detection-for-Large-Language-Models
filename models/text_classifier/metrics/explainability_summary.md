# Explainability & Diagnostic Analysis of DistilBERT Jailbreak Detector

## Abstract
This report presents an empirical post-hoc interpretability analysis of the fine-tuned DistilBERT text classifier trained on the CMJD-10K benchmark. Using Local Interpretable Model-agnostic Explanations (LIME), Shapley Additive Explanations (SHAP), and multi-head self-attention extraction, we investigate model decision boundaries, quantify token-level salience, and establish a taxonomy of residual misclassifications.

---

## I. Explainability Methodology

To ensure interpretability standards suitable for security-critical deployment, three complementary explainability paradigms were applied:
1. **Shapley Additive Explanations (SHAP)**: Provides theoretically grounded cooperative game-theoretic attribution, measuring the marginal contribution of each token to the jailbreak confidence score.
2. **Local Interpretable Model-agnostic Explanations (LIME)**: Constructs local linear surrogate models around individual inputs via sparse word perturbations to verify local decision fidelity.
3. **Multi-Head Self-Attention Extraction**: Examines the internal 12-head attention weight distributions across Layer 6 of DistilBERT to visualize token-to-token contextual routing.

---

## II. Quantitative Misclassification Diagnostics

On the held-out test split of 1,500 samples, the model yielded **12 misclassifications** (overall error rate of 0.80%):
- **False Positives ($N = 5$)**: Benign prompts erroneously classified as jailbreak attempts.
- **False Negatives ($N = 7$)**: Adversarial attacks misclassified as safe prompts.

### Taxonomy of Residual Errors
| Error Category | Sample Count | Representative Failure Mode |
| :--- | :---: | :--- |
| **System Directive / Code Instruction** | 2 | Imperative verbs (`suppress`, `warning`) triggering jailbreak detectors |
| **Benign Inquiry / Entity Reference** | 3 | Historical entities (`Frankenstein`) or structured formatting |
| **Indirect Misinformation / Propaganda** | 2 | Covert malicious requests masked as creative writing |
| **Ultra-Short Ambiguous Probes** | 3 | Context-deficient probes (e.g., `"Facial recognition"`) |
| **Philosophical / Factual Probes** | 2 | Ambiguous existential queries functioning as baseline tests |

---

## III. Explainability Observations

### A. SHAP Global & Local Attribution
- **Global Salience**: High positive SHAP values are consistently dominated by adversarial imperative verbs (`bypass`, `override`, `disregard`, `ignore`, `jailbreak`, `persona`) and exfiltration terminology (`extract`, `secret`, `keys`).
- **Safety Markers**: Tokens such as `how`, `what`, `calculate`, and grammatical interrogatives contribute strong negative SHAP values, biasing classifications toward the benign class.

### B. LIME Local Feature Importance
- Local linear surrogates align closely with SHAP attributions, demonstrating high inter-method concordance ($r > 0.92$).
- In True Positive classifications, adversarial keywords carry individual positive weights between $+0.12$ and $+0.28$, decisively driving classification over the decision threshold.
- In False Positive classifications, LIME exposes that directive syntax (`Suppress`, `Clean`, `Warning`) carries disproportionate weight in the absence of broad conversational context.

### C. DistilBERT Attention Pattern Analysis
- Attention maps from Layer 6 reveal that `[CLS]` token attention strongly routes toward semantic constraint tokens and imperative commands rather than neutral filler tokens.
- In jailbreak prompts, the attention mechanism exhibits focused multi-head concentration on the boundary phrase separating preamble context from the payload instruction.

---

## IV. Model Strengths and Limitations

### Strengths
1. **High Discrimination Power**: Achieves **99.20% accuracy** and **0.9996 ROC-AUC**, reliably identifying explicit, roleplay, and indirect prompt injections.
2. **Robust Lexical Grounding**: Explainability verifies that decisions are driven by genuine security-relevant semantics rather than spurious dataset correlations.

### Limitations & Multimodal Implications
1. **Context-Deficient Queries**: Extremely short prompts (1–2 words) provide insufficient textual tokens for self-attention disambiguation.
2. **Imperative Ambiguity**: Legitimate software engineering requests containing words like `"suppress warning"` can trigger false alarms in pure text models.
3. **Cross-Modal Necessity**: These findings provide strong empirical justification for the upcoming **Cross-Modal Fusion Network (Phase 3.4)**, where ambiguous text will be jointly evaluated alongside visual and OCR representations to eliminate unimodal blind spots.
