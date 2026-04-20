# Responsible AI Assignment 2

This repository contains a complete end-to-end workflow for building, stress-testing, auditing, mitigating, and operationalizing a toxicity classifier on Jigsaw-style data.

The work is split into five notebooks:
1. Part 1: Base model training and persistence
2. Part 2: Fairness and bias audit
3. Part 3: Adversarial robustness attacks
4. Part 4: Bias mitigation comparison and best-model selection
5. Part 5: Production guardrail pipeline with calibration and human review routing

The repository also includes a reusable production module:
- `pipeline.py`: Guardrail policy implementation used by Part 5

## Repository structure
- `part1.ipynb`
- `part2.ipynb`
- `part3.ipynb`
- `part4.ipynb`
- `part5.ipynb`
- `pipeline.py`
- `requirements.txt`
- `README.md`

## Project objective
The assignment demonstrates how to move from a baseline NLP classifier to a safer moderation system by:
- Measuring model performance and fairness by cohort
- Simulating realistic attacks (evasion and poisoning)
- Comparing mitigation techniques under fairness-accuracy trade-offs
- Deploying a layered moderation pipeline with confidence calibration and human review

## Data and cohort setup
Across parts, the workflow consistently:
- Uses the Jigsaw unintended-bias train CSV
- Drops null comments
- Creates a binary label: `toxic_label = 1 if toxic >= 0.5 else 0`
- Reconstructs identity columns including `lgbtq` (or fallback columns when needed)
- Uses fixed split sizes with reproducible seed:
	- `train_df`: 100,000
	- `eval_df`: 20,000
	- `random_state=42`

Key fairness cohorts used in analysis:
- High-black cohort: `black >= 0.5`
- Reference cohort: `black < 0.1 and white >= 0.5`

## What each part does

## Part 1 - Baseline training and model saving
Main outcomes:
- Prepares data and tokenizer (`distilbert-base-uncased`)
- Trains a DistilBERT sequence classifier with Hugging Face `Trainer`
- Evaluates core metrics (accuracy, macro-F1, ROC-AUC)
- Saves ROC/PR plots
- Saves model artifacts for reuse in later notebooks

Artifacts produced:
- Local model directory: `./saved_model`
- Optional Drive copy: `/content/drive/MyDrive/rai_assignment/saved_model`
- Threshold file: `threshold.txt` (default set to `0.4`)

Why it matters:
- Provides a stable baseline and avoids repeated retraining in Parts 2-5.

## Part 2 - Bias audit
Main outcomes:
- Reloads saved model and rebuilds eval split (no retraining)
- Scores toxicity probabilities on `eval_df`
- Computes cohort-level metrics:
	- TPR, FPR, FNR, precision
	- Disparate impact proxy via FPR ratio
- Computes AIF360 metrics:
	- Statistical Parity Difference (SPD)
	- Equal Opportunity Difference (EOD)
- Produces fairness visualizations and interpretation text

Why it matters:
- Quantifies whether one demographic cohort is over- or under-flagged compared to reference.

## Part 3 - Adversarial attacks
Implements two attack classes:

1) Character-level evasion attack
- Perturbs toxic text using zero-width spaces, homoglyph substitutions, and random char duplication
- Evaluates Attack Success Rate (ASR) on 500 high-confidence toxic samples
- Visualizes confidence shift before vs after perturbation

2) Label-flipping poisoning attack
- Flips exactly 5% training labels (5,000 when train size is 100,000)
- Retrains a fresh model on poisoned data
- Compares clean vs poisoned metrics on clean eval set
- Produces a grouped comparison plot

Artifacts produced:
- `attack1_results.png`
- `attack2_results.png`
- `./poisoned_model`

Why it matters:
- Demonstrates both test-time evasion risk and train-time data poisoning risk.

## Part 4 - Bias mitigation techniques
Part 4 is implemented as a self-sustained notebook (rebuilds data/splits and reloads model in-notebook).

Mitigations compared:
1. Reweighing (AIF360 preprocessing + weighted training loss)
2. Threshold optimization (Fairlearn post-processing, equalized odds)
3. Oversampling of high-black cohort in training data

Evaluation and selection:
- Compares each technique on:
	- Overall macro-F1
	- Cohort FPRs
	- SPD and EOD
- Builds a comparison table and visualization
- Uses a balance-style score to choose best practical trade-off
- Saves best trained mitigated model

Artifacts produced:
- `mitigation_comparison.png`
- `./reweighed_model`
- `./oversampled_model`
- `./best_mitigated_model`
- Drive copy: `/content/drive/MyDrive/jigsaw/best_mitigated_model`

Why it matters:
- Moves from audit to intervention and selects a candidate model for production use.

## Part 5 - Production guardrail pipeline
Part 5 is also self-sustained and production-oriented.

Pipeline design (3 layers):
1. Input filter: regex blocklist for severe categories (direct threats, self-harm directives, doxxing/stalking, dehumanization, coordinated harassment)
2. Model layer: calibrated toxicity probability with auto-block and auto-allow thresholds
3. Human review: uncertain middle band routed for manual moderation

Implemented work:
- Loads best mitigated model from Part 4 output
- Writes reusable module `pipeline.py`
- Fits isotonic calibrator and plots reliability before/after calibration
- Runs smoke tests covering all decision layers
- Runs 1,000-sample demo with:
	- Layer distribution pie chart
	- Auto-actioned subset metrics (F1/precision/recall)
	- Review queue composition
	- Blocklist category hit counts
- Performs threshold band sensitivity analysis:
	- narrow: 0.45-0.55
	- current: 0.40-0.60
	- wide: 0.30-0.70
- Outputs recommendation balancing safety, UX, and ops load

Drive outputs from Part 5:
- `calibration_before_after.png`
- `layer_distribution_pie.png`
- `blocklist_category_hits.png`
- `threshold_sensitivity.png`
- `threshold_sensitivity_table.csv`
- `pipeline.py` (copied to output folder)

## `pipeline.py` module summary
`pipeline.py` provides:
- `BLOCKLIST`: curated regex patterns by harm category
- `input_filter(text)`: hard-block detector
- `batch_predict_probs(...)`: batched model inference helper
- `ModerationPipeline`: layered decision class with:
	- `block_threshold` (default `0.6`)
	- `allow_threshold` (default `0.4`)
	- `predict(text)` returning structured decision payload

Decision schema:
- `decision`: `block`, `allow`, or `review`
- `layer`: source of decision (`input_filter`, `model_block`, `model_allow`, `human_review`)
- `category`: reason code
- `confidence`: confidence score

## How to run

## Recommended run order
1. Run `part1.ipynb` once to train and save baseline model.
2. Run `part2.ipynb` for fairness audit.
3. Run `part3.ipynb` for adversarial attacks.
4. Run `part4.ipynb` for mitigation comparison and best-model save.
5. Run `part5.ipynb` for production guardrail pipeline and final analysis.

Notes:
- Parts 4 and 5 are written to be self-sustained in a fresh runtime.
- They still require persisted artifacts from prior parts (saved baseline and best mitigated model paths).

## Environment setup
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For Colab execution:
- Mount Google Drive
- Ensure zipped CSVs are available in Drive
- Unzip to `/content/jigsaw`

Expected key paths:
- Data CSV: `/content/jigsaw/jigsaw-unintended-bias-train.csv`
- Baseline model: `/content/drive/MyDrive/rai_assignment/saved_model`
- Best mitigated model: `/content/drive/MyDrive/jigsaw/best_mitigated_model`

## Methods and metric references
Main libraries used:
- Hugging Face Transformers (`AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`)
- AIF360 (`Reweighing`, `ClassificationMetric`)
- Fairlearn (`ThresholdOptimizer`)
- scikit-learn (`IsotonicRegression`, `calibration_curve`, standard metrics)

Core metrics tracked across parts:
- Accuracy, macro-F1, ROC-AUC
- TPR, FPR, FNR, precision
- SPD, EOD
- ASR (attack success rate)
- Review queue rate and auto-actioned performance under threshold bands

## Reproducibility and implementation choices
- Deterministic sampling and splits use `random_state=42` where specified.
- Threshold defaults to `0.4` for toxicity classification unless overridden.
- Model persistence is used heavily to avoid unnecessary retraining and to keep part outputs consistent.
- Part 4 and Part 5 are intentionally implemented to avoid hidden runtime dependence on Part 1 variables.

## Troubleshooting
- Missing saved model error:
	- Run Part 1 save cell to create `saved_model`.
- Missing `best_mitigated_model` in Part 5:
	- Run Part 4 selection/save cell first.
- `google.colab` import warnings in local IDE:
	- Expected outside Colab; these cells are for Colab execution.
- AIF360/Fairlearn import issues:
	- Reinstall from `requirements.txt`, restart kernel, rerun from top.
- NumPy/Transformers runtime mismatch in notebooks:
	- Restart runtime after install cell, then rerun all cells in order.

## Final deliverable summary
This project delivers:
- A trained and reloadable toxicity model baseline
- A fairness audit with cohort and parity metrics
- Adversarial robustness assessment (evasion + poisoning)
- Three mitigation strategies with quantitative comparison
- A selected mitigated model saved for deployment
- A production-style, layered moderation pipeline with calibration, threshold sensitivity analysis, and operational reporting
