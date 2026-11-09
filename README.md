# CS781 Capstone

**Anshuman Raj · 240148 · `anshumanr24@iitk.ac.in` · Codabench `a_240148`**

In-house legal NLP, phases 1–7. Tokenizer and models are trained from scratch (no InLegalBERT, no public checkpoints, no LLM APIs for training).

Repo: https://github.com/AnshumaRaj/CS781-Capstone

## Layout

| Path | What it is |
|---|---|
| `code/Phase1_LegalLM.ipynb` … `Phase7_PCR.ipynb` | Viva notebooks for the **selected** model of each phase |
| `code/*.py` | Modules those notebooks import |
| `code/phase{1–7}/` | Source for every attempt of that phase (not Codabench zips) |
| `code/evaluators/` | Local zip checkers |
| `report/` | LaTeX, figures, PDF |

Hello IITK is `code/` + `report/` at the zip root: notebooks, selected modules, report. No `submission.zip` files.

## Selected code per phase

| Phase | Rank (9 Nov 2026) | Code | Architecture |
|---|---|---|---|
| 1 Legal LM | 55 | `code/Phase1_LegalLM.ipynb`, `code/phase1_models.py` | efficient, greedy + EOS |
| 2 NER | 51 | `code/Phase2_NER.ipynb`, `code/phase2_lner.py` | efficient full-FT + CRF, 2 seeds |
| 3 Roles | 39 | `code/Phase3_Roles.ipynb`, `code/phase3_rr_own.py` | efficient LoRMA + BiLSTM-CRF |
| 4 CJPE | 96 | `code/Phase4_CJPE.ipynb`, `code/phase4_own.py` | efficient partial + ending cue |
| 5 Statutes | 23 | `code/Phase5_Statutes.ipynb`, `code/phase5_fineblend.py` | TF-IDF ⊕ efficient windows |
| 6 Summary | 22 | `code/Phase6_Summary.ipynb`, `code/phase6_summ.py` | encdec extractive, 720 words |
| 7 PCR | 33 | `code/Phase7_PCR.ipynb`, `code/phase7_rerank.py` | 8k TF-IDF + cite/head |

```
pip install -r code/requirements.txt
jupyter notebook code/Phase1_LegalLM.ipynb
```

Keep `RUN_TRAINING = False` unless you are on a Kaggle T4 with the dataset and a Phase-1 checkpoint.
