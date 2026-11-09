# CS781 Capstone — code

**Anshuman Raj · 240148 · `anshumanr24@iitk.ac.in` · Codabench `a_240148`**

Layout required by Hello IITK: one folder per phase.

```
code/phase1/ … code/phase7/
report/CS781-CP-Anshuman-Raj-240148-report.pdf
```

| Open this | Phase | Selected architecture |
|---|---|---|
| `phase1/Phase1_LegalLM.ipynb` | 1 Legal LM | efficient (RoPE, GQA, SwiGLU), greedy + EOS |
| `phase2/Phase2_NER.ipynb` | 2 NER | efficient full-FT + CRF |
| `phase3/Phase3_Roles.ipynb` | 3 Rhetorical roles | efficient LoRMA + BiLSTM-CRF |
| `phase4/Phase4_CJPE.ipynb` | 4 Judgment prediction | efficient partial FT, windows + ending cue |
| `phase5/Phase5_Statutes.ipynb` | 5 Statute IDs | per-label head blended with TF-IDF |
| `phase6/Phase6_Summary.ipynb` | 6 Summarization | encdec extractive, 720-word budget |
| `phase7/Phase7_PCR.ipynb` | 7 Prior-case retrieval | 8k-word TF-IDF + cite/head fields |

Each `phaseN/` folder holds that phase’s source (and earlier modules it imports). No Codabench `submission.zip` files.

Keep `RUN_TRAINING = False` unless you are on a Kaggle T4 with the dataset and a Phase-1 checkpoint.

```
pip install -r phase1/requirements.txt
jupyter notebook phase1/Phase1_LegalLM.ipynb
```
