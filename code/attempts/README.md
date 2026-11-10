# All Codabench attempts (source only)

This folder is **GitHub only**. Hello IITK does not include it.

The official selected system for each phase is still the single notebook in `../phaseN/`. Prediction `submission.zip` files are not stored here (they are not required).

| Phase | Selected (notebook) | Scripts that built other attempts |
|---|---|---|
| 1 | `../phase1/Phase1_LegalLM.ipynb` | `legal_bpe.py`, `phase1_models.py`, `phase1_run.py`, `phase1_fastgen.py`, `phase1_mix.py`, `legal_lm.py` |
| 2 | `../phase2/Phase2_NER.ipynb` | `phase2_lner.py`, `phase2_train.py` |
| 3 | `../phase3/Phase3_Roles.ipynb` | `phase3_rr_own.py`, `phase3_rr.py`, `phase3_ensemble.py`, `phase3_tune.py`, `phase3_cache.py` |
| 4 | `../phase4/Phase4_CJPE.ipynb` | `phase4_own.py` plus gate / blend / SWA / BitFit helpers in this folder |
| 5 | `../phase5/Phase5_Statutes.ipynb` | `phase5_fineblend.py`, `phase5_tfidf.py`, `phase5_lsi.py`, and the other blend/ensemble scripts |
| 6 | `../phase6/Phase6_Summary.ipynb` | `phase6_summ.py` (`--extract` 720w is selected; LoRMA/KD paths are in the same file) |
| 7 | `../phase7/Phase7_PCR.ipynb` | `phase7_rerank.py` (selected), `phase7_lexv4.py`, `phase7_fields.py`, `phase7_pcr.py`, `phase7_ltr50.py`, … |

Kaggle `push_*` / `fetch_*` / kernel-maker scripts are omitted on purpose.
