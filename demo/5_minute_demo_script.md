# 5-Minute Executive Demo Script: Cross-Modal Jailbreak Detection

**Target Audience**: SAP Hackathon Judges, Executive Panel, Placement Evaluators  
**Total Target Time**: Exactly 5 Minutes  
**Prerequisites**: Backend running on port 8001, Frontend open at `http://127.0.0.1:5173/`.

---

## Timeline & Narration

### Minute 1: The Problem Hook (0:00 – 1:00)
- **Screen**: Start on the **Dashboard (Overview)** page (`/`).
- **Narration**:
  > "Hello everyone. Today, enterprises are deploying Multimodal LLMs across SAP Joule, customer service, and document workflows. But there is a massive security loophole: **Cross-Modal Prompt Injections**.
  > 
  > Attackers bypass traditional text guardrails by rendering malicious instructions directly inside images. Text filters can't see the image, and computer vision models trigger massive false alarms—up to 40% on harmless presentation slides.
  > 
  > Our solution is **CMJD**—a trimodal defense firewall that combines **DistilBERT**, **CLIP ViT-B/32**, **EasyOCR**, and our **Adaptive Gated Fusion Engine**. As you see on this dashboard, our system was evaluated on a hard 120-sample benchmark, achieving **98.33% accuracy**, **100% attack recall**, and reducing false positives from 40% down to **2.86%**."

---

### Minute 2: Text Jailbreak Defense (1:00 – 2:00)
- **Action**: Click **Text Scanner** in the sidebar.
- **Action**: Click the preset **"DAN 11.0 Persona Hijack"**, then click **Run DistilBERT Scan**.
- **Screen**: Risk Gauge swings to 99.8%, Red JAILBREAK badge flashes, SHAP attribution map displays.
- **Narration**:
  > "Let's first test our text modality. I'll select the infamous DAN—Do Anything Now—persona attack. 
  > 
  > Within 13 milliseconds, our fine-tuned DistilBERT identifies the exploit with **99.8% risk**, categorizing it as a Persona Hijack.
  > 
  > Notice this **SHAP Token Attribution Map**: it visually explains the neural reasoning. Red highlighted tokens like 'DAN', 'unrestricted', and 'rules' contributed heavily toward the attack verdict. This provides complete explainability for compliance."

---

### Minute 3: Solving the False-Positive Dilemma (2:00 – 3:15)
- **Action**: Click **Image Scanner** in the sidebar.
- **Action**: Click the preset **"Educational Slide (Benign)"**, then click **Run Visual Scan**.
- **Screen**: Shows green SAFE badge, 4.1% risk score, EasyOCR extracted text, and Grad-CAM attention heatmap.
- **Narration**:
  > "Now let's examine the primary reason vision models fail in production: the **false positive problem**.
  > 
  > Here is a standard university computer science slide on binary search trees and sorting algorithms. To a pure vision model, dense typography looks identical to an attack poster—a standard vision classifier would flag this as a threat with 96% risk!
  > 
  > But look at CMJD: EasyOCR extracts all 14 lines of typography, DistilBERT analyzes the computer science content, and our **Semantic Gating Rule** activates. It automatically discounts the spurious visual risk from 95.8% down to **4.1%**, giving a clear **SAFE** verdict. Legitimate enterprise workflows are never blocked!"

---

### Minute 4: Cross-Modal Synergistic Attack Neutralization (3:15 – 4:15)
- **Action**: Click **Cross-Modal Scanner** in the sidebar.
- **Action**: Click the preset **"Prompt Injection Poster (Threat)"**, then click **Run Cross-Modal Fusion**.
- **Screen**: Shows red JAILBREAK badge, 100% Risk Gauge, dynamic contribution weights bar, and decision reasoning text.
- **Narration**:
  > "Now, here is the ultimate challenge: a **Cross-Modal Synergistic Attack**.
  > 
  > The attacker uploads an adversarial poster instructing the LLM to ignore safety guidelines and dump system credentials.
  > 
  > Notice what happens: CLIP identifies visual adversarial styling, EasyOCR transcribes the concealed directives, and DistilBERT confirms the malicious intent. 
  > 
  > Our **Adaptive Contribution Bar** dynamically allocates 35% weight to vision and 65% to text. Because both modalities confirm elevated risk, our **Bimodal Consensus Override** triggers immediately, locking the risk at **100.0%** and blocking the query before it ever touches your LLM."

---

### Minute 5: Enterprise Alignment & Wrap-Up (4:15 – 5:00)
- **Action**: Click **Analytics** in the sidebar, briefly hover over the ablation table.
- **Action**: Click **Scan Reports**, show the audit log and point to the **Export CSV** and **Print/PDF** buttons.
- **Narration**:
  > "Looking at our analytics, our Adaptive Fusion delivers a **+21.66% accuracy uplift** over text alone and **+56.66%** over vision alone.
  > 
  > For enterprise deployment, CMJD is packaged as a high-throughput microservice ready for **SAP AI Core**, **SAP BTP Generative AI Hub**, and **SAP Joule**. It logs every scan with cryptographic audit trails ready for the EU AI Act.
  > 
  > Zero models were retrained, latency is under 100 milliseconds, and the system is production-ready today. Thank you!"
