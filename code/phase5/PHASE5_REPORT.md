# Phase 5 — Legal Statute Identification

**Deadline:** 26 Oct 2026, 11:59 PM IST  
**Task:** Multi-label statute ID prediction (100 labels, IDs 0–99)  
**Primary metric:** Macro-F1  **Secondary:** Micro-F1, subset accuracy  
**Constraint:** Start only from the four Phase 1 models. No public checkpoints, no LLM APIs.

## 1. Data

| Split | Documents | Labels |
|---|---|---|
| Train | 42,750 | given |
| Dev | 10,181 | given (model selection + thresholds) |
| Test | 13,019 | hidden |

About **3.7 statutes per document**. Rarest label has a few hundred train examples. Documents are long (median ~400–500 words, p90 ~2,800). Statute mentions in the text are masked.

## 2. Model

Each document is cut into up to **6 windows of 256 tokens** (head-to-tail if it fits, otherwise spread). The Phase 1 backbone encodes every window. A **per-label attention head** (query over all tokens) is added to a mean-pooled linear path whose bias is the label-frequency prior.

Loss is binary cross-entropy, sometimes with a positive-class weight. Predictions use **per-label thresholds** tuned on dev. If a document would be empty, the label with the highest score/threshold ratio is kept.

Honest scores in this report use 2-fold threshold tuning on halves of **dev that the model did not train on**.

## 3. Required experiments (all four Phase 1 models)

| # | Architecture | Adaptation | Why it counts | Honest dev Macro-F1 |
|---|---|---|---|---|
| 1 | **efficient** | **LoRMA** (compulsory) | PEFT 1 | 0.338 (2 ep) → **0.358** (4 ep) |
| 2 | **decoder** | Partial (last 8 blocks) | PEFT 2 | 0.247 |
| 3 | **hybrid** | Adapters (rank 64) | PEFT 3 | 0.205 |
| 4 | **encdec** | Distillation from efficient teacher | **Mandatory KD** | 0.225 |

Additional PEFT on the same models (not extra techniques): decoder LoRMA 0.157, hybrid partial 0.259, encdec partial 0.209.

## 4. What actually moved the score

Phase 1 models alone plateau near **0.36** Macro-F1 (best single: efficient full fine-tune, 0.366; 384-token windows, 0.371). Published BERT baselines on this task are ~0.59 / 0.65; those weights are forbidden.

A **TF-IDF + per-label logistic regression** model, trained only on competition text (no pretrained net), scores **0.411** on its own. Blending it with the efficient neural logits is the honest best:

| Submission | Honest Macro-F1 | File |
|---|---|---|
| TF-IDF + efficient LoRMA/partial ensemble | **0.422** | `p5_sub/efficient_fineblend_tfidf` |
| Decoder partial + TF-IDF | 0.409 | `p5_sub/decoder_partial_tfidf` |
| Encdec partial + TF-IDF | 0.408 | `p5_sub/encdec_partial_tfidf` |
| Hybrid adapter + TF-IDF | 0.407 | `p5_sub/hybrid_adapter_tfidf` |

Asymmetric loss, label-graph boosting, SVM, and extra seeds of the same full fine-tune did **not** beat 0.422.

## 5. Train-on-dev runs (not honest)

Full fine-tune on labeled train+dev reports **0.571** on the same dev. Hierarchical + train+dev reports **0.496**. Those numbers are leakage. The zips (`efficient_traindev`, `efficient_hierdev`) are valid test files and were the last attempt at a higher **hidden** test score. They must not be quoted as 0.5+ local evidence.

## 6. Analysis (against the spec)

- **PEFT vs full fine-tune:** LoRMA on efficient (~3M trainable) reached 0.358; unfreezing the whole efficient body (~110M) reached 0.366. Parameter-efficient methods kept almost all of that at much lower cost. Adapters and BitFit-style partial on decoder/hybrid lagged because those backbones are weaker generators of statute signal.
- **Long documents:** 6×256 windows were enough. 8–12 windows and 384-token windows added ~0.01 at most.
- **Rare labels / thresholds:** Per-label cuts beat a global cut by ~0.01–0.04 Macro-F1. Macro-F1 is dominated by rare statutes; class weighting helped a little, co-occurrence graphs did not.
- **Knowledge distillation:** The efficient teacher was the strongest neural model; the encdec student reached 0.225, above a from-scratch encdec partial (0.209), but far below the teacher. Logit matching did not close the architecture gap.
- **Phase 1 representations:** They are useful (neural models learn; F1 rises from ~0.05 to ~0.36) but, without a public legal BERT, bag-of-ngrams still wins Macro-F1. The blend is the practical system.
- **Cost:** Efficient LoRMA, 4 epochs, ~7.5 h on a Kaggle T4. Full fine-tune similar wall time, more memory. Laptop 6 GB OOMs on batch>2 or 10-window LoRMA.

## 7. What to upload

Four architectures, as required:

1. efficient — `p5_sub/efficient_fineblend_tfidf/submission.zip` (honest best) or `p5_sub/efficient_traindev/submission.zip` (hidden-test gamble)
2. decoder — `p5_sub/decoder_partial_tfidf/submission.zip`
3. encdec — `p5_sub/encdec_partial_tfidf/submission.zip`
4. hybrid — `p5_sub/hybrid_adapter_tfidf/submission.zip`
