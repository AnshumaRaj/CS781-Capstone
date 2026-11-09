# CS781 Capstone — code (starter layout)

**Anshuman Raj · 240148 · `anshumanr24@iitk.ac.in` · Codabench `a_240148`**

Each phase is one notebook, numbered like the official Phase-1 starter (markdown heading, then a short cell). The notebooks import the `.py` files in this same folder. They are **not** the course GPT-2 starter: tokenizer and models are trained from scratch.

| Open this | Phase | Selected architecture |
|---|---|---|
| `Phase1_LegalLM.ipynb` | 1 Legal LM | efficient (RoPE, GQA, SwiGLU), greedy + EOS |
| `Phase2_NER.ipynb` | 2 NER | efficient full-FT + CRF |
| `Phase3_Roles.ipynb` | 3 Rhetorical roles | efficient LoRMA + BiLSTM-CRF |
| `Phase4_CJPE.ipynb` | 4 Judgment prediction | efficient partial FT, windows + ending cue |
| `Phase5_Statutes.ipynb` | 5 Statute IDs | per-label head blended with TF-IDF |
| `Phase6_Summary.ipynb` | 6 Summarization | encdec extractive, 720-word budget |
| `Phase7_PCR.ipynb` | 7 Prior-case retrieval | 8k-word TF-IDF + cite/head fields |

Keep `RUN_TRAINING = False` unless you are on a Kaggle T4 with the dataset and a Phase-1 checkpoint.

Shared modules next to the notebooks:

- `legal_bpe.py` — Dice-weighted byte BPE, vocab 8192
- `phase1_models.py` — decoder, hybrid, efficient, encdec (`build_model` enforces 50M–300M)
- `phase2_lner.py` — LoRMA, CRF, character-aligned NER
- `phase3_rr_own.py` — sentence BiLSTM-CRF
- `phase4_own.py` — window classifier
- `phase5_tfidf.py`, `phase5_lsi.py`, `phase5_blend.py` — statute blend
- `phase6_summ.py` — extractive selector
- `phase7_rerank.py` — PCR first stage

`evaluators/` checks a `prediction.json` zip. Prediction zips themselves are not included.

```
pip install -r requirements.txt
jupyter notebook Phase1_LegalLM.ipynb
```
