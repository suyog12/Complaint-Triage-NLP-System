# Complaint Triage for Retirement Services: Automated Routing and Escalation Risk Scoring

## What this project is

A pension or retirement services provider receives thousands of written complaints a month
by email, web form, and regulator referral. Today a human reads each one, decides which team
owns it, and adds it to a queue in arrival order. Three things go wrong with that process:
complaints bounce between teams for days before anyone works them, a routine statement
question gets the same urgency as an allegation that a withdrawal was mishandled, and
regulated complaints carry hard response deadlines that turn into fines if they sit too long.

This project builds a model that reads a complaint's raw text and does two things:

1. **Routes it** to the team that should handle it (9 product categories).
2. **Scores its escalation risk**, i.e. how likely it is to end in monetary relief, so the
   highest-risk complaints can be prioritized instead of worked in arrival order.

## Why this data

Pension complaint data is not available in useful volume. This project trains on the CFPB
(Consumer Financial Protection Bureau) Consumer Complaint Database instead, which has 3.82
million usable narratives across nine product categories. The working assumption is that a
pension provider's complaint mix (transfers, withdrawals, statement errors, servicing delays)
is structurally similar enough to CFPB's categories that the routing and escalation approach
carries over. That is an assumption, not a proven result.

## Why these design choices

- **Balanced 54,000-row sample, not the full 3.82M rows.** The real population is 65% one
  class (Credit reporting). Training on that distribution teaches a model to guess Credit
  reporting and score well on raw accuracy while being useless for the other eight classes.
  6,000 narratives per class puts all nine on equal footing.
- **Deduplication before sampling.** The 54,000-row sample originally contained only 51,091
  unique narratives (some complaints, largely from credit repair firms, are filed as
  near-identical bulk copies). Left alone, duplicates would leak across the train/validation/
  test splits and inflate every result. Narratives are deduplicated by hash before splitting.
- **A frozen 70/15/15 split, reused everywhere.** Every model reads the exact same
  train/validation/test rows via `src/preprocess.py`, so results are comparable across models
  and nothing gets tuned on data it was also tested on.
- **No text cleaning.** CFPB redacts personal information as runs of `XXXX`. That pattern is
  left in the text (not stripped, lowercased, or stemmed) because it varies by product and
  can itself carry signal; the models decide what matters.
- **Three model families**, from cheapest to most expensive: Naive Bayes and Logistic
  Regression (TF-IDF features), and a fine-tuned BERT transformer. A fourth candidate, a
  hybrid that routes with Logistic Regression when it is confident and falls back to BERT
  otherwise, is also evaluated, since the point of comparing model families is to find the
  best deployment tradeoff, not just the highest single number.

## Key results

**Product routing** (test set, 8,100 complaints, macro-F1):

| Model | Macro-F1 | Notes |
|---|---:|---|
| Naive Bayes | 0.764 | |
| Logistic Regression | 0.810 | |
| BERT (fine-tuned) | 0.820 | ~29x Logistic Regression's train time |
| **Hybrid (Logistic Regression + BERT)** | **0.828** | ~3x BERT's throughput |

The hybrid routes with Logistic Regression when it is confident and falls back to BERT
otherwise. It beats fine-tuned BERT outright, not just on cost: a McNemar test on the paired
test predictions gives p = 0.0001, and a paired bootstrap puts the macro-F1 gain over BERT
alone at 0.008 with a 95% confidence interval of [0.004, 0.011] that does not cross zero.

**Escalation risk** (test set, 6.4% positive): Naive Bayes and Logistic Regression score below
a majority-class baseline on raw accuracy while catching far more true escalations; PR-AUC
(0.290 / 0.307 / 0.327 for Naive Bayes / Logistic Regression / BERT) is the metric that
actually reflects usefulness here. At each model's own cost-optimal threshold, BERT beats
Logistic Regression by about 9% on expected cost.

**Business simulation**: auto-routes 48.7% of complaints at 95.5% accuracy, and its
high-priority flag catches 67.1% of complaints that genuinely ended in monetary relief, at
24.3% precision (roughly 3.8x lift over the 6.4% base rate).

All of the numbers behind these results, plus the full per-class breakdowns and threshold
sweeps, are in `outputs/metrics/` and `outputs/figures/` (see below).

## Repository layout and what each file is for

```
Final Project/
├── run.py                          # single entrypoint - runs the entire pipeline
├── requirements.txt                # exact package versions for the conda environment used
├── README.md                       # this file
├── data/
│   └── processed/
│       └── splits.parquet          # 54,000-row deduplicated sample with a frozen train/val/test split
├── src/
│   ├── config.py                   # paths, random seed, product-name normalization map
│   ├── data.py                     # builds the balanced sample from the raw CFPB CSV
│   ├── preprocess.py               # builds the frozen split; every model loads data through here
│   ├── check_dupes.py              # verifies zero duplicate narratives and zero split leakage
│   ├── baselines.py                # Naive Bayes and Logistic Regression, both tasks
│   ├── learning_curve.py           # how much training data Logistic Regression actually needs
│   ├── transformer.py              # BERT/DistilBERT fine-tuning, both tasks, quick or full mode
│   ├── evaluate.py                 # builds every comparison table and figure, incl. the hybrid model
│   └── triage.py                   # the business simulation: routing + priority thresholds
├── notebooks/
│   └── bert_colab.ipynb            # self-contained BERT training notebook for Google Colab (GPU)
├── outputs/
│   ├── metrics/                    # every JSON/CSV result table this project produces
│   ├── figures/                    # every chart this project produces (PNG)
│   └── *.npy, *.csv                # raw per-document predictions and probabilities
└── models/                         # trained model files (.joblib), not tracked in this repo, see Setup
```

`data/raw/complaints.csv` (the ~8.6GB full CFPB export) and `data/processed/sample.parquet`
(an intermediate, pre-split artifact) are not tracked in this repository. Neither is needed to
run the project: everything downstream of sampling reads `data/processed/splits.parquet`,
which is tracked. See Setup for how to regenerate them if you want to.

### What each `src/` module actually does, and how to run it alone

Every module can be run directly with `python -m src.<module_name>` from the project root, or
through `run.py`, which runs all of them in the right order. Direct invocation is useful for
re-running one step without waiting for the whole pipeline.

- **`config.py`** — no runnable behavior. Defines the project's file paths, the random seed
  used everywhere (42), and the mapping that collapses CFPB's several historical names for
  the same product (e.g. "Credit reporting, credit repair services, or other personal
  consumer reports" and "Credit reporting or other personal consumer reports" both become
  "Credit reporting").
- **`data.py`** — `python -m src.data`. Reads `data/raw/complaints.csv`, filters to rows with
  a usable narrative and a company response, deduplicates by narrative hash, samples 6,000
  rows per product class, and writes `data/processed/sample.parquet`. Only needs to be run
  once, or if you intentionally want to rebuild the sample from scratch. Requires the raw CSV
  (see Setup).
- **`preprocess.py`** — `python -m src.preprocess`. Reads `sample.parquet`, creates the frozen
  70/15/15 stratified train/validation/test split, and writes `data/processed/splits.parquet`.
  Also exposes `load_split(name)`, which every other module uses to read train/val/test
  consistently. Only needs to be run once; `splits.parquet` is meant to be reused forever
  after, never regenerated mid-project.
- **`check_dupes.py`** — `python -m src.check_dupes`. Reads `splits.parquet` and reports how
  many duplicate narratives remain and whether any duplicate group spans more than one split
  (which would mean train/test leakage). Should report zero of both.
- **`baselines.py`** — `python -m src.baselines`. Trains Naive Bayes and Logistic Regression
  (TF-IDF features, 100,000 features, 1-2 grams) on both tasks (product routing, escalation
  risk), evaluates on test, and saves the trained models to `models/`, predictions to
  `outputs/`, and metrics to `outputs/metrics/baselines_product.json` and
  `baselines_escalation.json`.
- **`learning_curve.py`** — `python -m src.learning_curve` (single run) or
  `python -m src.learning_curve --multiseed` (3 seeds per size, slower but reports mean and
  standard deviation). Trains Logistic Regression on increasingly large subsets of the
  training data (1,000 up to the full 37,800) to check whether more training data would
  still help, or whether the sample size already chosen is past the point of diminishing
  returns. Writes `outputs/metrics/learning_curve.json` (and `learning_curve_multiseed.json`
  for the multi-seed version).
- **`transformer.py`** — `python -m src.transformer [--quick] [--task product|escalation|both]`.
  Fine-tunes a BERT-family model on both tasks. `--quick` uses DistilBERT on a 2,000-document
  subsample for a few minutes' CPU smoke test; the default (no flag) uses full
  `bert-base-uncased` on the complete 37,800-document training set, which needs a GPU (see
  Setup and Troubleshooting). Writes `outputs/metrics/bert_product.json` and
  `bert_escalation.json`, plus raw per-document predictions to `outputs/`.
- **`evaluate.py`** — `python -m src.evaluate`. Reads every metrics file produced above (and
  tolerates any that do not exist yet) and builds all the comparison tables and figures: product
  routing and escalation comparison tables, the prevalence-weighted accuracy table, the
  confusion matrix figure, the learning curve figure, and the hybrid Logistic Regression + BERT
  router (its own handoff threshold is tuned on validation separately from any other threshold
  in the project, then tested for statistical significance against BERT alone with a McNemar
  test and a paired bootstrap). Writes everything to `outputs/metrics/` and `outputs/figures/`.
- **`triage.py`** — `python -m src.triage`. Runs the actual business simulation: scores every
  test-set complaint for routing confidence and escalation risk, tunes both operating
  thresholds on validation under an explicit cost assumption, applies them once to test, and
  reports the operational numbers (what share auto-routes, at what accuracy; what share gets
  flagged high-priority, and what fraction of true escalations that flag catches). Also
  produces the BERT-vs-Logistic-Regression escalation comparison at each model's own
  cost-optimal threshold. Writes `outputs/triage_simulation.csv` and several files under
  `outputs/metrics/`.

### `notebooks/bert_colab.ipynb`

This notebook exists because full BERT fine-tuning (`bert-base-uncased`, all 37,800 training
documents, 2 epochs) needs a GPU; it takes roughly 15-20+ hours on CPU and about 15-20 minutes
on a free Google Colab T4 GPU. It duplicates the training logic in `transformer.py` in a
self-contained way (it does not import anything from `src/`, since Colab will not have this
project's package installed).

**How to use it:**

1. Open the notebook in Google Colab (upload it, or open it from Google Drive).
2. Set the runtime to a GPU: Runtime → Change runtime type → T4 GPU.
3. Upload `data/processed/splits.parquet` to the Colab session (the Files panel on the left;
   drag and drop, or use the upload icon). The notebook looks for it at `/content/splits.parquet`
   by default.
4. Run all cells (Runtime → Run all). It trains both tasks in sequence and prints progress as
   it goes.
5. When it finishes, the last cell zips everything it produced. Download that zip via the
   Colab file browser.
6. Unzip it locally and copy `bert_product.json` and `bert_escalation.json` into this
   project's `outputs/metrics/`, and the `*_bert_pred.npy` / `*_bert_val_pred.npy` /
   `*_bert_proba.npy` / `*_bert_val_proba.npy` files into `outputs/` (the project root of
   outputs, not the metrics subfolder).
7. Run `python -m src.evaluate` and `python -m src.triage` locally afterward to fold the new
   BERT results into the comparison tables, the hybrid router, and the triage simulation.

The notebook also runs locally (outside Colab) for a quick sanity check before uploading it:
launch it from the `notebooks/` directory or the project root with the `nlp-final` environment
and it finds `data/processed/splits.parquet` on its own. Set `QUICK_MODE = True` in the config
cell first, since full BERT on a CPU is not something you want to wait for locally.

Google Colab sessions lose everything in `/content/` when they disconnect or time out. If you
need the uploaded file to survive a disconnect, mount Google Drive instead (a commented option
is described in the notebook itself) rather than re-uploading each time.

## Setup

1. **Get the raw data (optional).** Download the full CFPB Consumer Complaint Database from
   `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`, unzip it, and place
   `complaints.csv` at `data/raw/complaints.csv`. This step is only needed if you want to
   rebuild `sample.parquet` and `splits.parquet` from scratch (`--rebuild-data`, below); the
   `splits.parquet` this project actually uses is already tracked in this repository, so most
   runs do not need the raw CSV at all.
2. **Create the environment.**
   ```bash
   conda create -n nlp-final python=3.11
   conda activate nlp-final
   pip install -r requirements.txt
   ```
3. **Windows only: fix a known conda/SSL issue before running anything that downloads a BERT
   model.** See Troubleshooting below. This is a one-time environment fix, not something this
   project's code does automatically.

## How to run everything

The whole pipeline runs through one entrypoint:

```bash
python run.py
```

This chains, in order: a data-readiness check, `check_dupes`, `baselines` (Naive Bayes +
Logistic Regression, both tasks), `learning_curve`, `transformer` (BERT, quick mode by
default), `evaluate`, and `triage`. Every step prints its own progress and a timing line when
it finishes, and the whole run ends with `done`.

**What to expect:** on a CPU-only machine with no GPU, the default run takes roughly 20-25
minutes end to end, verified by a full run (check_dupes 0.5s, baselines 2.1 min, learning_curve
2.2 min, BERT quick mode 20.5 min, evaluate 21.6s, triage 10.7s). BERT quick mode dominates the
runtime and its exact time varies noticeably with how loaded the machine is; every other step
combined takes under 5 minutes. It needs internet access the first time it runs, to download
the DistilBERT model weights from the Hugging Face Hub (cached locally after that, so
subsequent runs do not need internet for this step). At the end, `outputs/metrics/` and
`outputs/figures/` contain every table and chart summarized above, and
`outputs/triage_simulation.csv` contains the per-complaint business simulation output.

**Flags:**

- `--bert skip|quick|full` (default `quick`). `skip` leaves BERT out entirely and finishes in
  about 5 minutes; useful for re-running everything else without waiting on BERT, or if BERT
  results already exist in `outputs/` from a previous run (quick, full, or the Colab notebook)
  and you do not want them overwritten. `full` fine-tunes complete `bert-base-uncased` on the
  entire training set; this needs a GPU or expect 15-20+ hours on CPU (`run.py` prints a
  warning and proceeds anyway if no GPU is detected, since a genuine `--bert full` request is
  assumed to be intentional).
- `--rebuild-data`. Downloads the full CFPB CSV and rebuilds `sample.parquet` and
  `splits.parquet` from scratch. Skipped by default, since the split this project's results
  depend on is already tracked in this repository and must never be silently regenerated (a
  different split would make every result non-reproducible).

**Reproducing the exact full-BERT numbers above:** they were produced on a Colab T4 GPU via
`notebooks/bert_colab.ipynb`, not via `run.py`'s default quick mode, which is a CPU smoke test
that verifies the training code runs correctly, not a performance benchmark. Running
`python run.py` with its default `--bert quick` on a machine that already has full-mode BERT
results in `outputs/` will overwrite them with quick-mode results. Use `--bert skip` to run
everything else without touching existing BERT output, or re-run the Colab notebook (see
above) to regenerate the full-mode numbers.

**Running one step at a time** (useful for iterating without re-running the whole pipeline):
every module in `src/` can also be run directly, listed above under each module's description,
for example `python -m src.baselines` or `python -m src.triage`.

## What is in `outputs/`

- **`outputs/metrics/*.json`, `*.csv`** — every numeric result this project produces. This
  includes per-model comparison tables for both tasks (`product_comparison.csv`,
  `escalation_comparison.csv`), the raw metrics each model produced
  (`baselines_product.json`, `baselines_escalation.json`, `bert_product.json`,
  `bert_escalation.json`), the learning curve (`learning_curve.json`,
  `learning_curve_multiseed.json`), the prevalence-weighted accuracy table
  (`prevalence_weighted_accuracy.csv`), every threshold sweep this project ran
  (`routing_threshold_sweep.csv`, `priority_threshold_sweep.csv`,
  `hybrid_threshold_sweep.csv`, `bert_priority_threshold_sweep.csv`), the hybrid-vs-BERT
  significance test result (`hybrid_vs_bert_significance.json`), and the triage business
  simulation summary (`triage_summary.json`).
- **`outputs/figures/*.png`** — every chart this project produces: the confusion matrix, the
  learning curve, and the threshold sweep plots (routing, priority, hybrid handoff).
- **`outputs/*.npy`** — raw per-document predictions and probabilities for every model on
  every split that needed them (e.g. `product_logistic_regression_pred.npy`,
  `escalation_bert_proba.npy`). These are what `evaluate.py` and `triage.py` read to build the
  tables above; they are not meant to be read directly, but are kept so any comparison in this
  project can be recomputed or audited without retraining anything.
- **`outputs/triage_simulation.csv`** — one row per test-set complaint, with its predicted
  routing, routing confidence, escalation risk, and the two operational decisions (auto-route
  vs. manual review, standard vs. high priority) the business simulation made for it.
- **`models/`** — trained Naive Bayes and Logistic Regression models and their TF-IDF
  vectorizers, saved as `.joblib` files, so `evaluate.py` and `triage.py` can score new data
  without retraining. Not tracked in this repository (regenerated by `baselines.py` in a few
  seconds); BERT model weights are not saved anywhere, since they are large and can be
  regenerated from `transformer.py` or the Colab notebook if needed.

## Troubleshooting

### `huggingface_hub` download fails on Windows conda environments

**Symptom:** `python -m src.transformer` (or `run.py`, which calls it) fails before training
starts, with a traceback ending in something like:

```
FileNotFoundError: [Errno 2] No such file or directory
...
ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
```

**Cause:** the conda `openssl` package's activation script sets the `SSL_CERT_FILE` environment
variable to `$CONDA_PREFIX/ssl/cacert.pem`, a Unix-style path. On Windows, conda installs the
certificate bundle at `$CONDA_PREFIX/Library/ssl/cacert.pem` instead, so `SSL_CERT_FILE` points
at a file that does not exist. Older `requests`-based code silently ignores a missing
`SSL_CERT_FILE` and falls back to `certifi`. The `transformers` / `huggingface_hub` versions
this project uses call `httpx` for Hub downloads, and `httpx` raises immediately when the file
named by `SSL_CERT_FILE` does not exist, so every model download fails before training starts.

**Fix:** point `SSL_CERT_FILE` at the certificate bundle that actually exists. From an activated
`nlp-final` environment:

```bash
conda env config vars set SSL_CERT_FILE="$CONDA_PREFIX/Library/ssl/cacert.pem"
conda deactivate
conda activate nlp-final
```

(On Windows PowerShell, `$CONDA_PREFIX` resolves the same way once the environment is active.)

This only needs to be done once per environment. If you rebuild the `nlp-final` environment
from scratch, redo it before running `src.transformer`, `run.py`, or anything else that
downloads from the Hugging Face Hub.

### `splits.parquet not found`

`run.py` and most `src/` modules refuse to run without `data/processed/splits.parquet`, since
regenerating it would silently produce a different train/test split and invalidate every
result. If you see this, either make sure `data/processed/splits.parquet` is present (it is
tracked in this repository, so a normal clone already has it), or run
`python run.py --rebuild-data` if you specifically intend to rebuild it from the raw CFPB CSV.

## Measured timings (CPU, no GPU)

Measured from a full, verified end-to-end run of `python run.py` (default flags):

| Step | Time |
|---|---:|
| `check_dupes.py` | under 1 second |
| `baselines.py`, both tasks | ~2 minutes |
| `learning_curve.py` (single run) | ~2.2 minutes |
| `transformer.py --quick`, both tasks | ~20.5 minutes |
| `evaluate.py` | ~22 seconds |
| `triage.py` | ~11 seconds |
| **`run.py` default (`--bert quick`)** | **~25 minutes total** |
| `run.py --bert skip` | ~5 minutes total |

BERT quick mode dominates the default runtime and its exact time varies with machine load; an
earlier, less-loaded run measured it at ~12 minutes instead of ~20.5, so treat this as "expect
it to be the majority of the runtime," not a precise figure. Full-mode BERT (`--bert full`)
needs a GPU; without one, expect 15-20+ hours per task on CPU, which is why
`notebooks/bert_colab.ipynb` exists.
