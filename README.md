# CS781 Capstone

**Anshuman Raj · 240148 · `anshumanr24@iitk.ac.in` · Codabench `a_240148`**

In-house legal NLP, phases 1–7. Tokenizer and models are trained from scratch (no InLegalBERT, no public checkpoints, no LLM APIs for training).

Repo: https://github.com/AnshumaRaj/CS781-Capstone

## Layout

| Path | What it is |
|---|---|
| `code/Phase1_LegalLM.ipynb` … `Phase7_PCR.ipynb` | Viva notebooks (starter cell layout) |
| `code/*.py` | Modules the notebooks import |
| `code/best_submissions/` | **One selected Codabench zip per phase** (Hello IITK uses only these) |
| `code/all_submissions/` | Every other Codabench / local-run zip from the competition |
| `code/evaluators/` | Local zip checkers |
| `report/` | LaTeX, figures, PDF |

The Hello IITK zip is `code/` + `report/` at the zip root and contains **only** the seven best zips. Historical attempts live on this GitHub repo.

## Selected model per phase

| Phase | Rank (9 Nov 2026) | Zip | Architecture |
|---|---|---|---|
| 1 Legal LM | 55 | `code/best_submissions/phase1_submission.zip` | efficient, greedy + EOS |
| 2 NER | 51 | `code/best_submissions/phase2_submission.zip` | efficient full-FT + CRF, 2 seeds |
| 3 Roles | 39 | `code/best_submissions/phase3_submission.zip` | efficient LoRMA + BiLSTM-CRF |
| 4 CJPE | 96 | `code/best_submissions/phase4_submission.zip` | efficient partial + ending cue |
| 5 Statutes | 23 | `code/best_submissions/phase5_submission.zip` | TF-IDF ⊕ efficient windows |
| 6 Summary | 22 | `code/best_submissions/phase6_submission.zip` | encdec extractive, 720 words |
| 7 PCR | 33 | `code/best_submissions/phase7_submission.zip` | 8k TF-IDF + cite/head |

```
pip install -r code/requirements.txt
jupyter notebook code/Phase1_LegalLM.ipynb
```

Keep `RUN_TRAINING = False` unless you are on a Kaggle T4 with the dataset and a Phase-1 checkpoint.
