# Machine Learning & AI Theory Viva Questions: 25 In-Depth Questions & Answers

**Project**: Cross-Modal Jailbreak Detection for Large Language Models (CMJD)  
**Focus**: Deep learning, transformers, contrastive vision embeddings, attention mathematics, loss functions, and XAI derivations.

---

### 1. Derive the self-attention mechanism in the DistilBERT transformer encoder.
**Answer**:  
For an input embedding sequence $\mathbf{X} \in \mathbb{R}^{N \times d}$, self-attention projects $\mathbf{X}$ into Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) matrices via learned parameter weights $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$:
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$
The scaled dot-product attention computes compatibility between queries and keys:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
The scaling factor $1/\sqrt{d_k}$ prevents the dot products from growing excessively large for large values of $d_k$ (where $d_k = 768 / 12 = 64$), which would otherwise push the softmax function into regions with vanishingly small gradients.

---

### 2. How does multi-head attention differ from single-head attention?
**Answer**:  
Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, multi-head attention linearly projects queries, keys, and values $h$ times ($h=12$ in DistilBERT) with different learned projections to $d_k$ dimensions:
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$$
where $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$.
This allows the model to jointly attend to information from different representation subspaces at different positions—for instance, one head attends to imperative verbs (`ignore`), while another tracks syntax pronouns (`you`).

---

### 3. How was DistilBERT trained? Explain knowledge distillation.
**Answer**:  
DistilBERT was trained by distilling knowledge from a larger teacher network (`bert-base-uncased`, 110M parameters) into a smaller student network (6 layers, 66M parameters) using a triple loss objective:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{ce}} + \beta \mathcal{L}_{\text{mlm}} + \gamma \mathcal{L}_{\text{cos}}$$
1. **Distillation Cross-Entropy Loss ($\mathcal{L}_{\text{ce}}$)**:
   $$\mathcal{L}_{\text{ce}} = -\sum_i p_i^{\text{teacher}}(T) \log p_i^{\text{student}}(T)$$
   where softmax probabilities are smoothed by temperature parameter $T$ ($p_i = \exp(z_i/T)/\sum \exp(z_j/T)$). This forces the student to learn the full probability distribution (the "dark knowledge") of the teacher.
2. **Masked Language Modeling Loss ($\mathcal{L}_{\text{mlm}}$)**: Standard hard-target cross-entropy on masked tokens.
3. **Cosine Embedding Loss ($\mathcal{L}_{\text{cos}}$)**: Aligns the directions of student and teacher hidden state vectors.

---

### 4. What is the loss function used to fine-tune DistilBERT for jailbreak detection?
**Answer**:  
We fine-tuned DistilBERT using **Binary Cross-Entropy Loss with Logits** ($\mathcal{L}_{\text{BCE}}$):
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \sigma(z_i) + (1 - y_i) \log (1 - \sigma(z_i)) \right]$$
where $y_i \in \{0, 1\}$ is the ground-truth label, $z_i$ is the raw unnormalized logit output for the jailbreak class, and $\sigma(z) = 1/(1 + e^{-z})$ is the standard logistic sigmoid function. In PyTorch, combining the sigmoid and cross-entropy into `BCEWithLogitsLoss` provides numerical stability via the log-sum-exp trick.

---

### 5. What optimizer and learning rate schedule were used for DistilBERT fine-tuning?
**Answer**:  
- **Optimizer**: **AdamW** (Adam with decoupled weight decay):
  $$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$
  with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1 \times 10^{-8}$, and weight decay $\lambda = 0.01$.
- **Learning Rate Schedule**: Initial peak learning rate $\eta = 2.0 \times 10^{-5}$ with a **linear warmup** over the first 10% of total training steps, followed by a **linear decay** down to 0.

---

### 6. Explain the concept of Early Stopping and why it triggered at Epoch 2 for DistilBERT.
**Answer**:  
Early stopping is a regularization technique that halts training when validation performance ceases to improve, preventing overfitting on the training distribution. We configured early stopping with:
- Metric: Validation $F_1$-score (`val_f1`)
- Patience: 2 epochs
- Min delta: $0.001$
- **Epoch 1**: Val loss = 0.0889, Val $F_1$ = 0.9720
- **Epoch 2**: Val loss = 0.0416, Val $F_1$ = 0.9920 (Best checkpoint saved)
- **Epoch 3**: Val loss = 0.0533, Val $F_1$ = 0.9906 (Validation loss rose, $F_1$ dropped)
- **Epoch 4**: Val loss = 0.0615, Val $F_1$ = 0.9919 (Patience of 2 exhausted &rarr; training terminated at Ep. 4).

---

### 7. What is contrastive learning, and how was CLIP trained?
**Answer**:  
OpenAI's CLIP (Contrastive Language-Image Pretraining) was trained on a batch of $N$ image-text pairs $\{(\mathcal{I}_i, \mathcal{T}_i)\}_{i=1}^N$.
Let $\mathbf{v}_i \in \mathbb{R}^D$ be the normalized visual embedding from the Vision Transformer, and $\mathbf{u}_j \in \mathbb{R}^D$ be the normalized text embedding from the text transformer.
The model maximizes the cosine similarity $\mathbf{v}_i^T \mathbf{u}_i$ of the $N$ correct pairs while minimizing the similarity $\mathbf{v}_i^T \mathbf{u}_j$ ($i \ne j$) of the $N^2 - N$ incorrect pairs using a symmetric InfoNCE loss:
$$\mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathbf{v}_i^T \mathbf{u}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{v}_i^T \mathbf{u}_j / \tau)}$$
$$\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\mathbf{u}_i^T \mathbf{v}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{u}_i^T \mathbf{v}_j / \tau)}$$
$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} (\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})$$
where $\tau$ is a learned temperature parameter.

---

### 8. Explain how a Vision Transformer (ViT-B/32) processes an image.
**Answer**:  
1. **Patch Partitioning**: An image $\mathcal{I} \in \mathbb{R}^{H \times W \times C}$ (where $H=W=224, C=3$) is divided into non-overlapping patches of size $P \times P$ ($P=32$). The number of patches is $N = (224/32) \times (224/32) = 7 \times 7 = 49$.
2. **Linear Projection**: Each flattened patch $\mathbf{x}_p^i \in \mathbb{R}^{32 \times 32 \times 3} = \mathbb{R}^{3072}$ is linearly projected into dimension $D = 768$: $\mathbf{z}_0^i = \mathbf{x}_p^i \mathbf{E}$.
3. **Class Token & Position Embeddings**: A learnable classification token $\mathbf{x}_{\text{class}}$ is prepended, and 1D learnable position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are added:
   $$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{z}_0^1; \dots; \mathbf{z}_0^N] + \mathbf{E}_{\text{pos}}$$
4. **Transformer Encoder**: The sequence passes through 12 transformer encoder blocks with multi-head self-attention and MLP blocks.
5. **Output Projection**: The class token from the final layer is linearly projected into the 512-dim multimodal space.

---

### 9. Why did you use Batch Normalization in the CLIP MLP head?
**Answer**:  
Batch Normalization computes the mini-batch mean $\mu_B$ and variance $\sigma_B^2$:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
In our 3-layer MLP head ($512 \rightarrow 256 \rightarrow 64 \rightarrow 2$), Batch Normalization prevents internal covariate shift during training, allows faster convergence with a higher learning rate ($\eta = 10^{-3}$), and provides subtle regularization that prevents the head from overfitting on specific visual image resolutions.

---

### 10. Mathematically define the Matthews Correlation Coefficient (MCC) and explain why it is superior to Accuracy.
**Answer**:  
The Matthews Correlation Coefficient is defined from the confusion matrix:
$$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$
MCC ranges from $-1$ (complete disagreement) through $0$ (no better than random prediction) to $+1$ (perfect prediction). 
Unlike raw Accuracy—which can be misleading on imbalanced datasets (e.g. predicting 100% negative on a 99% negative dataset yields 99% accuracy but MCC = 0)—MCC produces a high score only if the prediction performs well across all four confusion matrix categories (TP, TN, FP, FN). Our DistilBERT model achieved an exceptional **MCC of 0.9840**.

---

### 11. Define ROC-AUC and PR-AUC. In what scenario is PR-AUC preferable?
**Answer**:  
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Measures the integral of the True Positive Rate ($\text{TPR} = \text{TP}/(\text{TP}+\text{FN})$) plotted against the False Positive Rate ($\text{FPR} = \text{FP}/(\text{FP}+\text{TN})$) across all classification thresholds $\tau \in [0, 1]$.
- **PR-AUC (Precision-Recall Area Under Curve)**: Measures Precision ($\text{TP}/(\text{TP}+\text{FP})$) plotted against Recall ($\text{TP}/(\text{TP}+\text{FN})$).
- **Preference**: In real-world enterprise traffic where adversarial attacks are rare events (e.g. 1 attack per 10,000 queries, severe class imbalance), ROC-AUC can present an overly optimistic picture because the large number of TNs keeps FPR small. PR-AUC focuses strictly on the minority positive class and is preferred for highly imbalanced anomaly detection.

---

### 12. Derive the Shapley value formulation used in SHAP.
**Answer**:  
Shapley values originate from cooperative game theory, dividing a total payout among players based on their marginal contributions. For a model prediction $f(x)$ and feature set $F$, the attribution $\phi_i$ of feature $i$ is:
$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
where:
- $S$ is a subset of features excluding feature $i$.
- $|S|!(|F| - |S| - 1)! / |F|!$ is the combinatorial probability weight representing the likelihood of coalition $S$ forming in a random permutation.
- $f(S \cup \{i\}) - f(S)$ is the marginal contribution of feature $i$ when added to coalition $S$.
SHAP satisfies four fundamental axiomatic properties: **Efficiency** ($\sum \phi_i = f(x) - \mathbb{E}[f(x)]$), **Symmetry**, **Dummy player**, and **Additivity**.

---

### 13. How does Kernel SHAP approximate exact Shapley values?
**Answer**:  
Computing exact Shapley values requires evaluating $2^{|F|}$ coalition subsets, which is computationally intractable for sequences with dozens of tokens. Kernel SHAP approximates $\phi_i$ by casting it as a weighted linear regression:
$$\arg\min_{\phi_0, \phi} \sum_{z' \in Z} \left[ f(h_x(z')) - \left(\phi_0 + \sum_{j=1}^M \phi_j z_j'\right) \right]^2 \pi_x(z')$$
where $z' \in \{0, 1\}^M$ is a binary coalition vector, $h_x(z')$ maps $z'$ back to original input space by replacing absent features with reference background values, and $\pi_x(z')$ is the Shapley kernel:
$$\pi_x(z') = \frac{M - 1}{\binom{M}{|z'|} |z'| (M - |z'|)}$$

---

### 14. How does LIME construct its local explanation surrogate?
**Answer**:  
LIME (Local Interpretable Model-agnostic Explanations) defines an explanation for instance $x$ as:
$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
where:
- $G$ is the class of interpretable linear models: $g(z') = w_g^T z'$.
- $\Omega(g)$ measures the complexity of explanation $g$ (e.g. limiting the number of non-zero features).
- $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$ is an exponential kernel measuring proximity between original sample $x$ and perturbed sample $z$.
- $\mathcal{L}$ is the squared loss weighted by proximity $\pi_x$:
  $$\mathcal{L}(f, g, \pi_x) = \sum_{z, z' \in \mathcal{Z}} \pi_x(z) (f(z) - g(z'))^2$$

---

### 15. Derive Grad-CAM for the CLIP Vision Transformer.
**Answer**:  
Let $y^c$ be the raw score (logit) for the JAILBREAK class $c$, and let $A_{i, j}^k$ be the activation at spatial position $(i, j)$ in feature map $k$ of the final self-attention layer (Layer 11).
1. **Gradient Computation**: Compute the partial derivative of $y^c$ with respect to $A_{i, j}^k$: $\frac{\partial y^c}{\partial A_{i, j}^k}$.
2. **Neuron Importance Weights**: Global average pool the gradients over width and height:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
3. **Rectified Linear Combination**: Weight the forward activation maps by $\alpha_k^c$ and apply ReLU:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
ReLU ensures the heatmap highlights only features that have a positive influence on the JAILBREAK class score.

---

### 16. What is the difference between Grad-CAM and raw Self-Attention rollout in ViTs?
**Answer**:  
- **Self-Attention Rollout**: Multiplies attention weight matrices across successive layers assuming linear combinations of token identities. However, attention rollout is **class-agnostic**—it shows what the transformer attends to in general, but cannot explain why the model chose `JAILBREAK` instead of `SAFE`.
- **Grad-CAM**: Uses class-specific backpropagated gradients to weight the activations. It specifically highlights the visual regions responsible for increasing the `JAILBREAK` score versus features supporting the `SAFE` score.

---

### 17. How does Dropout prevent overfitting during training?
**Answer**:  
During each training forward pass, Dropout randomly deactivates each neuron with probability $p$ (e.g., $p=0.2$ in DistilBERT, $p=0.3$ in CLIP MLP):
$$r_j \sim \text{Bernoulli}(1 - p), \quad \tilde{\mathbf{y}} = \mathbf{r} \odot \mathbf{y}$$
During backward propagation, gradients only propagate through active neurons. 
Mathematically, Dropout approximates an ensemble of $2^N$ sub-networks with shared weights. It prevents co-adaptation among neurons, forcing each neuron to learn robust, generalized representations that do not rely on the presence of specific co-occurring neurons. During inference, dropout is disabled and weights are scaled by $(1 - p)$.

---

### 18. What is the difference between Early Fusion, Late Fusion, and Gated Fusion?
**Answer**:  
- **Early Fusion**: Concatenates raw feature vectors $\mathbf{z} = [\mathbf{e}_{\text{img}} \mathbin{\Vert} \mathbf{e}_{\text{text}}]$ at the input or intermediate layers before passing through a joint classifier. Weakness: Cannot dynamically reweight when one modality is corrupted or missing.
- **Late Fusion**: Models compute predictions independently, and outputs are aggregated via static voting or linear averaging: $R = \alpha R_v + (1-\alpha)R_t$. Weakness: Static weights fail when typography causes extreme spurious scores in one modality.
- **Gated Fusion (Our Approach)**: Dynamically conditions modality contributions on input-dependent meta-features (visual entropy, text length, OCR confidence), adjusting effective weights and applying non-linear overrides.

---

### 19. Why does the bimodal risk distribution in your analytics show sharp peaks near 0% and 100%?
**Answer**:  
This bimodality is a hallmark of a well-calibrated, high-margin classifier. In an un-optimized model, predictions cluster ambiguously near the 50% threshold. In CMJD:
- Safe samples (educational slides, clean code, scenic images) trigger semantic gating and entropy damping, collapsing risk scores to $0\%\text{--}15\%$.
- Confirmed adversarial injections activate the bimodal consensus override, snapping risk scores to $95\%\text{--}100\%$.
This wide margin of separation ($> 80\%$) provides high operational confidence and robust thresholding.

---

### 20. What is catastrophic forgetting, and how did your architecture prevent it?
**Answer**:  
Catastrophic forgetting occurs when a neural network trained on Task A is fine-tuned on Task B, causing the network's weights to overwrite representations essential for Task A.
If we had fine-tuned CLIP end-to-end on our 2,000-image jailbreak dataset, CLIP would have lost its broad general zero-shot understanding of the visual world. We prevented this by **freezing the entire CLIP ViT-B/32 backbone** ($\nabla_\theta \mathcal{L} = 0$), only updating the lightweight 3-layer MLP projection head.

---

### 21. How do you evaluate whether DistilBERT is overfitting to specific jailbreak trigger keywords?
**Answer**:  
1. **LIME Perturbation Testing**: We perturbed trigger words with character typos (e.g. `d-a-n`, `0verride`, `byp@ss`). LIME showed that the model maintains high attention on the broader syntactic structure and semantics rather than exact keyword strings.
2. **Held-Out Generalization**: The model was tested on held-out splits containing novel paraphrased jailbreaks never seen during training, maintaining 99.20% accuracy.

---

### 22. Explain how EasyOCR's CRNN recognizer transcribes character sequences.
**Answer**:  
Once CRAFT identifies a bounding box crop, the CRNN (Convolutional Recurrent Neural Network) pipeline processes the cropped text line:
1. **CNN Feature Extractor**: Standard VGG/ResNet extracts high-level spatial visual feature sequences.
2. **Bidirectional LSTM**: A 2-layer BiLSTM captures long-range character context in both left-to-right and right-to-left directions.
3. **CTC (Connectionist Temporal Classification) Loss**: CTC maps the recurrent output frames to character labels without requiring pre-segmented character alignments, handling variable-width letters and spaces via a blank token $\epsilon$.

---

### 23. What is the mathematical justification for using quadratic damping on visual entropy?
**Answer**:  
In our entropy formula:
$$R_v' = R_v \cdot \left(\frac{H(\mathcal{I})}{H_{\text{max}}}\right)^2$$
A linear scaling factor $(H/H_{\text{max}})$ would still leave significant residual risk on low-entropy canvases (e.g., $H=1.5 \implies 1.5/8.0 = 0.1875 \implies 95\% \times 0.1875 = 17.8\%$ risk).
Squaring the ratio $(H/H_{\text{max}})^2$ introduces a steeper non-linear decay:
$$(1.5 / 8.0)^2 = (0.1875)^2 \approx 0.035 \implies 95\% \times 0.035 = 3.3\% \text{ risk}$$
This suppresses spurious visual activations to below our safety threshold while preserving sensitivity on complex natural scenes where $H > 6.0$ bits.

---

### 24. What is the difference between Precision and Recall in a cybersecurity guardrail? Which one should be prioritized?
**Answer**:  
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Percentage of flagged queries that were genuine attacks. High precision means minimal false alarms.
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Percentage of actual attacks that were successfully intercepted. High recall means zero missed breaches.
- **Priority**: In enterprise AI security, **Recall must be strictly prioritized** ($100.0\%$). A single missed jailbreak can leak customer databases or trigger unauthorized code execution. However, precision must remain high enough ($> 95\%$) to prevent operational paralysis. CMJD achieves **100.0% Recall** while maintaining **96.15% Precision**.

---

### 25. How do you handle multi-language prompt injections in CMJD?
**Answer**:  
1. EasyOCR natively supports 80+ language scripts (Latin, Cyrillic, Chinese, Arabic) by loading corresponding language dictionaries.
2. In Phase 4, DistilBERT can be swapped with `xlm-roberta-base` (multilingual RoBERTa), which processes 100 languages. In our current English configuration (`['en']`), DistilBERT also exhibits cross-lingual transferability on Romance languages due to subword tokenization (WordPiece).
