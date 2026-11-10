# CS781 Capstone

**Anshuman Raj · 240148 · `anshumanr24@iitk.ac.in` · Codabench `a_240148`**

Repo: https://github.com/AnshumaRaj/CS781-Capstone

Hello IITK and this repo share the **same selected notebook per phase**. Every other Codabench attempt lives only here, as source (not prediction zips).

```
code/phase1/Phase1_LegalLM.ipynb     ← selected (starter-style)
… 
code/phase7/Phase7_PCR.ipynb
code/attempts/phase1/ … phase7/      ← all other runs (GitHub only)
report/CS781-CP-Anshuman-Raj-240148-report.pdf
```

| Phase | Rank (9 Nov 2026) | Selected notebook | What it is |
|---|---|---|---|
| 1 Legal LM | 55 | [`code/phase1/Phase1_LegalLM.ipynb`](code/phase1/Phase1_LegalLM.ipynb) | efficient, greedy + EOS |
| 2 NER | 51 | [`code/phase2/Phase2_NER.ipynb`](code/phase2/Phase2_NER.ipynb) | efficient full-FT + CRF |
| 3 Roles | 39 | [`code/phase3/Phase3_Roles.ipynb`](code/phase3/Phase3_Roles.ipynb) | LoRMA + BiLSTM-CRF |
| 4 CJPE | 96 | [`code/phase4/Phase4_CJPE.ipynb`](code/phase4/Phase4_CJPE.ipynb) | partial FT + ending cue |
| 5 Statutes | 23 | [`code/phase5/Phase5_Statutes.ipynb`](code/phase5/Phase5_Statutes.ipynb) | TF-IDF blend |
| 6 Summary | 22 | [`code/phase6/Phase6_Summary.ipynb`](code/phase6/Phase6_Summary.ipynb) | encdec extractive 720w |
| 7 PCR | 33 | [`code/phase7/Phase7_PCR.ipynb`](code/phase7/Phase7_PCR.ipynb) | 8k TF-IDF + cite/head |

Weights (`.pt`), Codabench zips, and Kaggle push scripts are not in this repo. See [`code/attempts/README.md`](code/attempts/README.md) for the other runs.

```
pip install -r code/requirements.txt
jupyter notebook code/phase1/Phase1_LegalLM.ipynb
```
