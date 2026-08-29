import argparse
import sys
import time
import urllib.request
import zipfile

import torch

from src import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"


def _download_raw_csv():
    if C.RAW_CSV.exists():
        print(f"{C.RAW_CSV} already exists, skipping download")
        return
    C.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    zip_path = C.RAW_CSV.parent / "complaints.csv.zip"
    print(f"downloading {CFPB_URL} -> {zip_path} (~800MB, this takes a while)")
    urllib.request.urlretrieve(CFPB_URL, zip_path)
    print(f"unzipping {zip_path} -> {C.RAW_CSV.parent} (~8.6GB)")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(C.RAW_CSV.parent)
    zip_path.unlink()
    print(f"wrote {C.RAW_CSV}")


def _step(name, fn):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    t0 = time.time()
    fn()
    print(f"[{name} done in {time.time() - t0:.1f}s]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="download the 8.6GB CFPB CSV and regenerate sample.parquet/splits.parquet from "
             "scratch. Skipped by default since splits.parquet ships with the project.",
    )
    parser.add_argument(
        "--bert", choices=["skip", "quick", "full"], default="quick",
        help="quick: DistilBERT smoke test, ~12 min on CPU. full: bert-base-uncased on the "
             "full 37,800-doc train split, needs a GPU or ~15+ hours on CPU. skip: no BERT step.",
    )
    args = parser.parse_args()

    C.MODELS.mkdir(parents=True, exist_ok=True)
    (C.OUTPUTS / "metrics").mkdir(parents=True, exist_ok=True)
    (C.OUTPUTS / "figures").mkdir(parents=True, exist_ok=True)

    from src import data, preprocess, check_dupes, baselines, learning_curve, transformer, evaluate, triage

    if args.rebuild_data:
        _step("download + build sample", lambda: (_download_raw_csv(), data.build_sample()))
        _step("preprocess (frozen splits)", preprocess.build_splits)
    elif not C.SPLITS_PARQUET.exists():
        print(f"{C.SPLITS_PARQUET} not found and --rebuild-data was not passed.\n"
              "Either make sure data/processed/splits.parquet ships with the project, or "
              "run again with --rebuild-data.")
        sys.exit(1)

    _step("check_dupes", check_dupes.check)

    _step("baselines", lambda: (
        baselines.run_task("product", "product"),
        baselines.run_task("escalated", "escalation"),
    ))

    _step("learning_curve", learning_curve.run)

    if args.bert == "skip":
        print("\nskipping BERT (--bert skip)")
    else:
        cfg = transformer.QUICK if args.bert == "quick" else transformer.FULL
        if args.bert == "full" and not torch.cuda.is_available():
            print("\nWARNING: --bert full with no GPU detected. bert-base-uncased on the full "
                  "37,800-doc train split takes roughly 15-20+ hours on CPU (measured ~18h "
                  "estimate from a single-epoch CPU timing). This is what Colab/notebooks/"
                  "bert_colab.ipynb is for. Proceeding anyway since --bert full was explicit.")
        _step(f"transformer --{args.bert}", lambda: (
            transformer.run_task("product", cfg),
            transformer.run_task("escalation", cfg),
        ))

    _step("evaluate", evaluate.run)
    _step("triage", triage.run)

    print("\ndone")


if __name__ == "__main__":
    main()
