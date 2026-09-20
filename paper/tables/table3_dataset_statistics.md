# Table III: Dataset Partitioning, Class Balance & Taxonomic Distributions

Summary of datasets utilized across the textual, visual, and cross-modal evaluation stages.

| Dataset Corpus | Modality | Total Samples | Safe Samples | Jailbreak Samples | Partitions (Train / Val / Test) | Primary Attack Categories |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **CMJD-10K** | Pure Text Prompts | 10,000 | 5,000 (50%) | 5,000 (50%) | 7,000 / 1,500 / 1,500 (70/15/15) | DAN persona hijack, Developer mode override, Roleplay exploit, Base64 cipher |
| **CMJD-Vision** | Images (Photographic & Visual) | 2,000 | 1,000 (50%) | 1,000 (50%) | 1,400 / 300 / 300 (70/15/15) | Typographic injections, Stylized posters, System warning dialogs, Terminal payloads |
| **IEEE Challenge Set** | Multimodal Pairings (Image + Text) | 120 | 70 (58.3%) | 50 (41.7%) | Held-Out Benchmark Only | Camouflaged low-contrast text, Noisy OCR, Educational lecture slides, UI screenshots |

### Taxonomic Class Distribution in CMJD-10K
- **Direct Instruction Overrides**: 28.4% (e.g. "Ignore previous directives...")
- **Persona Hijacking (DAN / evil-bot)**: 24.1% (e.g. "You are now unrestricted...")
- **Roleplay / Creative Fiction Exploits**: 21.3% (e.g. "In a hypothetical script...")
- **Obfuscation / Encoding (Base64/Ciphers)**: 14.2% (e.g. "SWdub3Jl...")
- **Logic / Reverse Psychology Traps**: 12.0% (e.g. "Explain why you cannot answer X...")
