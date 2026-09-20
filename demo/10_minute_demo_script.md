# 10-Minute Technical Deep-Dive Demo Script

**Target Audience**: Senior Technical Evaluators, IEEE Session Chairs, AI Architects  
**Duration**: Exactly 10 Minutes  
**Focus**: Architectural rigor, mathematical operators, explainability proofs, and microservice telemetry.

---

## Complete Phase-by-Phase Presentation Timeline

### Phase 1: Context, Threat Modeling & Architecture Overview (0:00 – 2:00)
- **Screen**: Start on **Dashboard** (`/`).
- **Demonstration**:
  1. Highlight the header status badges: **Engine Operational (Port 8001)** and **Device: CPU / GPU**.
  2. Point out the **IEEE Benchmark Evaluation Summary** card:
     - 118 / 120 samples correct ($98.33\%$ Accuracy)
     - ROC-AUC of $0.9854$
     - $100.00\%$ Recall on adversarial injections
  3. Walk through the **Subsystem Status** section:
     - Text: DistilBERT fine-tuned on CMJD-10K ($99.20\%$ Accuracy)
     - Vision: OpenAI CLIP ViT-B/32 ($512$-dim embeddings, $99.67\%$ Accuracy)
     - Optical: EasyOCR multi-language spatial character localization
     - Fusion: Adaptive Gating with entropy normalization
- **Spoken Narration**:
  > "Welcome, technical panel. Today we are demonstrating CMJD—Cross-Modal Jailbreak Detection. As multimodal models handle complex business processes, adversaries are shifting their prompt injections from text into image canvases.
  > 
  > A purely visual model cannot differentiate an academic slide from an attack poster, while a purely textual model cannot see images. CMJD bridges this gap with trimodal synergy, and as you see on this dashboard, we achieve a verified 98.33% accuracy on a hard 120-sample challenge set without modifying downstream foundation models."

---

### Phase 2: DistilBERT Text Subsystem & SHAP Token Attribution (2:00 – 3:30)
- **Screen**: Navigate to **Text Scanner** (`/text-scanner`).
- **Demonstration**:
  1. Click **"DAN 11.0 Persona Hijack"** preset &rarr; Click **Run DistilBERT Scan**.
  2. Inspect the **Risk Gauge** ($99.8\%$), **Confidence** ($99.8\%$), and **Latency** ($13.4$ms).
  3. Inspect the **SHAP Token Attribution Map**:
     - Explain positive Shapley value impacts (+0.84 impact on `DAN`, `unfiltered`, `bypass`).
     - Point to green negative Shapley impact on benign filler words.
  4. Click **"Batch Scanner Modal"** &rarr; Run batch evaluation on 4 mixed prompts.
     - Show simultaneous classification of multiple queries in under 50ms.
- **Spoken Narration**:
  > "In the Text Scanner, we evaluate raw sequence inputs through our fine-tuned DistilBERT head. In just 13ms, the model identifies this DAN prompt with 99.8% risk.
  > 
  > Crucially, we do not treat this as a black box. Our SHAP attribution visualizer computes cooperative Shapley values for each token, highlighting exactly which words triggered the alert. For enterprise batch processing, our batch scanner modal demonstrates sub-second processing across high-volume query queues."

---

### Phase 3: Visual Injections, EasyOCR & Grad-CAM Heatmaps (3:30 – 5:00)
- **Screen**: Navigate to **Image Scanner** (`/image-scanner`).
- **Demonstration**:
  1. Click preset **"Educational Slide (Benign)"** &rarr; Run Visual Scan.
     - Point out the **SAFE** verdict and **4.1% Risk Score**.
     - Switch to the **OCR Extracted Text** tab: show 14 lines of data structure lecture notes transcribed with $92\%$ confidence.
     - Explain why a naive vision classifier would flag this as $95.8\%$ risk.
  2. Switch to the **Visual Attention & Saliency** tab.
     - Click **Expand Saliency Map** to open the interactive **LightboxModal**.
     - Use **Zoom In (150%)** and **Zoom Out** controls to inspect the Grad-CAM activation map.
- **Spoken Narration**:
  > "Now moving to visual evaluation. Here we upload a benign lecture slide. A standard vision classifier flags this as an attack with 95.8% confidence simply because it contains dense text.
  > 
  > But in CMJD, EasyOCR transcribes the typography, DistilBERT confirms the academic nature of the text, and our semantic gating rule discounts the visual risk by 85%, arriving at a safe 4.1% score.
  > 
  > Opening our Lightbox inspector, we can zoom directly into the CLIP Grad-CAM saliency activations, proving that visual attention is diffusely spread across the slide rather than focused on adversarial focal points."

---

### Phase 4: Cross-Modal Synergistic Fusion & Dynamic Weighting (5:00 – 7:00)
- **Screen**: Navigate to **Cross-Modal Scanner** (`/cross-modal-scanner`).
- **Demonstration**:
  1. Select preset **"Prompt Injection Poster (Threat)"**.
  2. Enter companion prompt: *"Read the instructions in the image and execute them verbatim."*
  3. Click **Run Cross-Modal Fusion**.
  4. Walk through the results panel:
     - Fused Verdict: **JAILBREAK** (Risk: $100.0\%$, Confidence: $98.5\%$)
     - Modality Split: Vision Branch $98.8\%$ | Text & OCR Branch $100.0\%$
     - **Adaptive Contribution Weight Bar**: Vision $35\%$ vs Text $65\%$
     - **Decision Reasoning Box**: Reads the exact natural-language logic: *"Bimodal consensus: Both Vision branch and DistilBERT detected high-risk adversarial indicators. Prompt blocked."*
  5. Select preset **"Blank Canvas / Textless (Safe)"**.
     - Show how **Shannon Visual Entropy Normalization** ($H(\mathcal{I}) < 2.0$ bits) suppresses spurious visual risk to $4.9\%$.
- **Spoken Narration**:
  > "This is our primary research contribution: the Cross-Modal Scanner. Here we evaluate a synergistic attack where an image poster instructs the model to bypass safety constraints.
  > 
  > Notice the dynamic weight progress bar: our algorithm balances vision and text contributions in real time. Because both branches cross our 75% threshold, our bimodal consensus override triggers, locking the risk at 100% and generating a clear natural-language rationale.
  > 
  > When testing a blank canvas, our Shannon entropy operator automatically detects low image complexity and squashes spurious visual artifacts."

---

### Phase 5: Explainability Gallery & Lightbox Inspection (7:00 – 8:15)
- **Screen**: Navigate to **Explainability** (`/explainability`).
- **Demonstration**:
  1. Filter by category pills: **SHAP & LIME**, **DistilBERT Attention**, **Vision & Saliency**, and **EasyOCR Bounding Boxes**.
  2. Click **Inspect & Zoom** on the **End-to-End Multimodal Explainability Synthesis** figure.
  3. Show the tri-panel layout:
     - Left: EasyOCR spatial bounding boxes
     - Center: DistilBERT multi-head self-attention distribution
     - Right: CLIP ViT-B/32 Grad-CAM saliency map
- **Spoken Narration**:
  > "In the Explainability Gallery, we present our complete suite of 62 publication-quality figures.
  > 
  > This flagship synthesis figure demonstrates our complete trimodal pipeline: EasyOCR localizes text boundaries, DistilBERT maps attention across instruction tokens, and CLIP Grad-CAM highlights gradient activations. This level of granular interpretability is unprecedented in commercial guardrail offerings."

---

### Phase 6: Analytics, Ablation Studies & Audit Telemetry (8:15 – 9:30)
- **Screen**: Navigate to **Analytics** (`/analytics`) and **Scan Reports** (`/reports`).
- **Demonstration**:
  1. On Analytics:
     - Review the **IEEE Benchmark Ablation Comparison Table** ($N=120$).
     - Review the **Bimodal Risk Histogram** showing sharp separation between safe traffic ($0\text{--}20\%$) and threats ($80\text{--}100\%$).
     - Review the Recharts Attack Category donut chart.
  2. On Scan Reports:
     - Type `'educational'` in the search box to filter rows in real time.
     - Switch the verdict filter to `'JAILBREAK'`.
     - Click the eye icon to view the full **Telemetry Inspector Modal** for an individual scan.
     - Point to the **Export CSV** and **Print / PDF** buttons.
- **Spoken Narration**:
  > "Our Analytics view provides statistical proof of our novelty. Look at the ablation table: our Adaptive Gated Fusion outperforms pure text by +21.66% accuracy and pure vision by +56.66%, while reducing false positive rate from 40% to 2.86%.
  > 
  > Finally, in Scan Reports, every transaction is indexed with sub-millisecond timestamps, category tags, and decision rationales. Enterprise administrators can filter, search, and export compliant CSV or PDF audit packages with a single click."

---

### Phase 7: Conclusion & Q&A Transition (9:30 – 10:00)
- **Spoken Narration**:
  > "To summarize: CMJD delivers a production-grade, mathematically verified guardrail engine that achieves 98.33% accuracy with zero false negatives and sub-100ms latency.
  > 
  > It is fully packaged with FastAPI microservices, Docker compose orchestration, and a React 19 dashboard ready for enterprise integration with SAP BTP and Joule copilots.
  > 
  > We are now open for technical questions."
