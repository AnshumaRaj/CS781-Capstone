# How to publish the GitHub repo and the Hello IITK zip

Deadline: **9 November 2026, 11:59 PM IST**.

Your identifiers (already filled in the report):

- Name: **Anshuman Raj**
- Roll: **240148**
- Email: **anshumanr24@iitk.ac.in**
- Codabench: **a_240148**
- Zip name: `CS781-CP-Anshuman-Raj-240148-report_code.zip`
- PDF name: `CS781-CP-Anshuman-Raj-240148-report.pdf`

## 0. One thing you must still paste

**GitHub URL.** The report uses `https://github.com/anshumanr24/CS781-Capstone`. If your GitHub username is different, change `\githuburl` at the top of `Report.tex` and recompile.

Ranks in Table 1 are already filled from the Codabench board of 9~November~2026: **55, 51, 39, 96, 23, 22, 33** (phases 1–7). Official primary metrics of the selected zips: 0.8328, 0.5446, 0.529, 0.6763, 0.4252, 0.8503, 0.4519.

## 1. Create the public GitHub repository

On a machine with `git` and a GitHub account:

```powershell
cd "C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\final_submit\github_repo"
git init
git add .
git commit -m "CS781 capstone: in-house legal NLP, phases 1-7"
gh repo create CS781-Capstone --public --source . --remote origin --push
```

If `gh` is not logged in, create an empty **public** repo named `CS781-Capstone` on github.com, then:

```powershell
git remote add origin https://github.com/YOURUSER/CS781-Capstone.git
git branch -M main
git push -u origin main
```

Check in the browser that the repo is **public**, that `README.md` renders, and that **no** `kaggle.json`, `.env`, or API token is in the tree (`git grep -i token` should be empty).

Do **not** upload model weights, the Hugging Face dump, or Kaggle outputs. They are too large and not required. The README tells a re-implementer where those files live on a local disk.

## 2. Recompile the PDF after the URL and ranks are set

```powershell
cd "C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\final_submit\report"
pdflatex Report.tex
bibtex Report
pdflatex Report.tex
pdflatex Report.tex
```

Confirm **at most 10 pages** before the LLM Usage Declaration. References and the declaration may spill to extra pages. Copy the PDF to:

`report/CS781-CP-Anshuman-Raj-240148-report.pdf`

(the pack script below does that rename).

## 3. Build the Hello IITK zip (root must be `code/` and `report/` only)

```powershell
python "C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\final_submit\pack_helloiitk.py"
```

That writes:

`C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\final_submit\CS781-CP-Anshuman-Raj-240148-report_code.zip`

Unzip it to a temp folder and confirm you see **exactly**:

```
code/
report/
```

and **not** `final_submit/code`. Upload that zip on Hello IITK.

## 4. What is already selected as the best model per phase

| Phase | Code to point TAs at | Selected zip (local copy) |
|---|---|---|
| 1 | `code/phase1/phase1_models.py`, `phase1_fastgen.py` | `phase1/local_runs/best_08328/` (rank 55) |
| 2 | `code/phase2/phase2_lner.py` | `efficient_s42_s7_ensemble` (rank 51) |
| 3 | `code/phase3/phase3_rr_own.py` | `efficient_lorma_full_ce2_rarity` (rank 39) |
| 4 | `code/phase4/phase4_own.py` | efficient partial + ending cue (rank 96) |
| 5 | `code/phase5/phase5_final_blend.py` | `efficient_fineblend_tfidf` (rank 23) |
| 6 | `code/phase6/phase6_summ.py` | `encdec_extract_720-score(.3988)` (rank 22) |
| 7 | `code/phase7/phase7_rerank.py` | `rerank/submission.zip` (rank 33) |

Keep the Phase 7 **rerank** zip as the Codabench selection. Do not replace it with pair-encoder or 12k lexical zips.
