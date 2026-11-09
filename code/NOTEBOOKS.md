# Viva notebooks

These notebooks follow the official Phase-1 starter layout: numbered markdown headings, short cells, one idea per cell.

They describe **our** system, not the starter GPT-2 pipeline.

| Notebook | What you explain |
|---|---|
| `Phase1_LegalLM.ipynb` | Dice BPE, four from-scratch LMs, efficient GQA, greedy+EOS |
| `Phase2_NER.ipynb` | Character-aligned BPE, LoRMA, CRF, full-FT ensemble |
| `Phase3_Roles.ipynb` | Sentence pool + BiLSTM-CRF, rarity bias |
| `Phase4_CJPE.ipynb` | Multi-window packing, ending cue, partial FT |
| `Phase5_Statutes.ipynb` | Per-label heads, thresholds, TF-IDF blend |
| `Phase6_Summary.ipynb` | EncDec extractive 720-word selector |
| `Phase7_PCR.ipynb` | 8k TF-IDF first stage, cite/head fields |

Open from this folder **or** from `code/viva/` in the GitHub / Hello-IITK zip. Cells import `phase1_models.py`, `legal_bpe.py`, and the task scripts sitting next to them.

`RUN_TRAINING = False` on purpose. Architecture cells use `inspect.getsource` so you can talk through the real classes without allocating 118M weights.
