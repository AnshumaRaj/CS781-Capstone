# Source for every phase attempt

Codabench `submission.zip` files are **not** in this repo. Each `phaseN/` folder is the Python that produced those attempts.

| Folder | Selected module (also copied next to the viva notebook) |
|---|---|
| `phase1/` | `phase1_models.py` / `phase1_fastgen.py` (efficient, greedy+EOS) |
| `phase2/` | `phase2_lner.py` (efficient full-FT + CRF, two seeds) |
| `phase3/` | `phase3_rr_own.py` (efficient LoRMA + BiLSTM-CRF) |
| `phase4/` | `phase4_own.py` (efficient partial + ending cue) |
| `phase5/` | `phase5_fineblend.py` (TF-IDF ⊕ efficient windows) |
| `phase6/` | `phase6_summ.py` (encdec extractive, 720 words) |
| `phase7/` | `phase7_rerank.py` (8k TF-IDF + cite/head) |
