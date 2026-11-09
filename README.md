# CS781 Capstone

**Anshuman Raj · 240148 · `anshumanr24@iitk.ac.in` · Codabench `a_240148`**

In-house legal NLP, phases 1–7. Tokenizer and models are trained from scratch.

Repo: https://github.com/AnshumaRaj/CS781-Capstone

```
code/phase1/ … code/phase7/
report/CS781-CP-Anshuman-Raj-240148-report.pdf
```

| Phase | Rank (9 Nov 2026) | Notebook | Selected code |
|---|---|---|---|
| 1 Legal LM | 55 | `code/phase1/Phase1_LegalLM.ipynb` | `phase1_models.py` efficient, greedy + EOS |
| 2 NER | 51 | `code/phase2/Phase2_NER.ipynb` | `phase2_lner.py` full-FT + CRF |
| 3 Roles | 39 | `code/phase3/Phase3_Roles.ipynb` | `phase3_rr_own.py` LoRMA + BiLSTM-CRF |
| 4 CJPE | 96 | `code/phase4/Phase4_CJPE.ipynb` | `phase4_own.py` partial + ending cue |
| 5 Statutes | 23 | `code/phase5/Phase5_Statutes.ipynb` | `phase5_fineblend.py` TF-IDF blend |
| 6 Summary | 22 | `code/phase6/Phase6_Summary.ipynb` | `phase6_summ.py` encdec extractive 720w |
| 7 PCR | 33 | `code/phase7/Phase7_PCR.ipynb` | `phase7_rerank.py` 8k TF-IDF + cite/head |

Each `code/phaseN/` folder is **only the selected run**: one notebook plus the `.py` files it imports. No other attempts, no Codabench zips.

```
pip install -r code/phase1/requirements.txt
jupyter notebook code/phase1/Phase1_LegalLM.ipynb
```
