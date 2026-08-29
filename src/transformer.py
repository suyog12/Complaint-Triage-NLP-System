import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    average_precision_score, roc_auc_score,
)
from src import config as C
from src.preprocess import load_split

QUICK = {
    "mode": "quick",
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "epochs": 1,
    "train_docs": 2000,
    "test_docs": 1000,
    "batch_size": 16,
}

FULL = {
    "mode": "full",
    "model_name": "bert-base-uncased",
    "max_length": 256,
    "epochs": 2,
    "train_docs": None,
    "test_docs": None,
    "batch_size": 16,
}


class NarrativeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _prepare_labels(task_name, tr, te):
    if task_name == "product":
        classes = sorted(tr["product"].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_tr = tr["product"].map(class_to_idx).to_numpy()
        y_te = te["product"].map(class_to_idx).to_numpy()
    else:
        classes = [0, 1]
        class_to_idx = {0: 0, 1: 1}
        y_tr = tr["escalated"].to_numpy()
        y_te = te["escalated"].to_numpy()
    return classes, class_to_idx, y_tr, y_te


def _encode_labels(task_name, df, class_to_idx):
    col = "product" if task_name == "product" else "escalated"
    return df[col].map(class_to_idx).to_numpy()


def _run_eval(model, loader, device):
    model.eval()
    all_preds, all_probs = [], []
    forward_seconds = 0.0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels")
            tf0 = time.time()
            outputs = model(**batch)
            forward_seconds += time.time() - tf0
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
    end_to_end_seconds = time.time() - t0
    return np.concatenate(all_preds), np.concatenate(all_probs), end_to_end_seconds, forward_seconds


def _decode_predictions(task_name, preds_idx, idx_to_class):
    if task_name == "product":
        return np.array([idx_to_class[i] for i in preds_idx], dtype=object)
    return preds_idx.astype(int)


def run_task(task_name, cfg, train_seed=None, test_seed=None, write_json=True, write_predictions=True):
    # train_seed drives model init, DataLoader shuffling, and which training docs get
    # subsampled - varying it is how the seed sweep measures training-side instability.
    # test_seed is held fixed across a sweep so eval is on the same docs every time and the
    # measured variance isn't partly just test-set-composition noise.
    train_seed = C.SEED if train_seed is None else train_seed
    test_seed = C.SEED if test_seed is None else test_seed
    torch.manual_seed(train_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    tr = load_split("train")
    va = load_split("val")
    te = load_split("test")
    if cfg["train_docs"]:
        tr = tr.sample(n=min(cfg["train_docs"], len(tr)), random_state=train_seed).reset_index(drop=True)
    if cfg["test_docs"]:
        va = va.sample(n=min(cfg["test_docs"], len(va)), random_state=test_seed).reset_index(drop=True)
        te = te.sample(n=min(cfg["test_docs"], len(te)), random_state=test_seed).reset_index(drop=True)

    classes, class_to_idx, y_tr, y_te = _prepare_labels(task_name, tr, te)
    y_va = _encode_labels(task_name, va, class_to_idx)
    num_labels = len(classes)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    print(f"\n=== {task_name} | {cfg['mode']} | model={cfg['model_name']} | "
          f"train={len(tr)} val={len(va)} test={len(te)} device={device} ===")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=num_labels
    ).to(device)

    train_loader = DataLoader(
        NarrativeDataset(tr.narrative, y_tr, tokenizer, cfg["max_length"]),
        batch_size=cfg["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        NarrativeDataset(va.narrative, y_va, tokenizer, cfg["max_length"]),
        batch_size=cfg["batch_size"], shuffle=False,
    )
    test_loader = DataLoader(
        NarrativeDataset(te.narrative, y_te, tokenizer, cfg["max_length"]),
        batch_size=cfg["batch_size"], shuffle=False,
    )

    class_weights = None
    if task_name == "escalation":
        n_pos = max(int((y_tr == 1).sum()), 1)
        n_neg = int((y_tr == 0).sum())
        class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=device)
        print(f"class weights: n_train_docs={len(tr)} (subsampled, not the full split) "
              f"n_pos={n_pos} n_neg={n_neg} weight_tensor={class_weights.tolist()}")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    if class_weights is not None:
        matches = torch.equal(loss_fn.weight.cpu(), class_weights.cpu())
        print(f"loss_fn.weight={loss_fn.weight.tolist()} matches computed tensor: {matches}")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    model.train()
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        running_loss, n_steps = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            n_steps += 1
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        print(f"epoch {epoch + 1} mean loss={running_loss / n_steps:.4f}")
    train_seconds = time.time() - t0

    preds, probs, end_to_end_seconds, forward_seconds = _run_eval(model, test_loader, device)
    val_preds, val_probs, _, _ = _run_eval(model, val_loader, device)
    n_test = len(te)

    if write_predictions:
        # Raw per-document predictions, not just aggregate metrics - needed for anything that
        # scores individual documents downstream: threshold sweeps (val probabilities) and a
        # hybrid LogReg+BERT routing comparison (test predictions), neither of which can be
        # done from a classification_report alone.
        np.save(C.OUTPUTS / f"{task_name}_bert_pred.npy", _decode_predictions(task_name, preds, idx_to_class))
        np.save(C.OUTPUTS / f"{task_name}_bert_val_pred.npy", _decode_predictions(task_name, val_preds, idx_to_class))
        if task_name == "escalation":
            np.save(C.OUTPUTS / f"{task_name}_bert_proba.npy", probs[:, 1])
            np.save(C.OUTPUTS / f"{task_name}_bert_val_proba.npy", val_probs[:, 1])

    accuracy = accuracy_score(y_te, preds)
    macro_f1 = f1_score(y_te, preds, average="macro")
    target_names = [str(idx_to_class[i]) for i in range(num_labels)]
    report = classification_report(y_te, preds, target_names=target_names, output_dict=True, zero_division=0)

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_seconds": round(train_seconds, 2),
        "end_to_end_seconds": round(end_to_end_seconds, 3),
        "predict_seconds": round(forward_seconds, 3),
        "inference_n_docs": n_test,
        "train_n_docs": len(tr),
        "train_seed": train_seed,
        "test_seed": test_seed,
        "report": report,
        "mode": cfg["mode"],
        "model_name": cfg["model_name"],
        "epochs": cfg["epochs"],
        "max_length": cfg["max_length"],
    }
    if task_name == "escalation":
        pos_probs = probs[:, 1]
        result["pr_auc"] = average_precision_score(y_te, pos_probs)
        result["roc_auc"] = roc_auc_score(y_te, pos_probs)
        result["positive_f1"] = f1_score(y_te, preds, pos_label=1, zero_division=0)

    docs_per_sec = n_test / end_to_end_seconds if end_to_end_seconds else float("nan")
    print(f"{task_name:12s} acc={accuracy:.4f}  macroF1={macro_f1:.4f}  "
          f"train={train_seconds:.1f}s  end-to-end={docs_per_sec:.1f} docs/sec"
          + (f"  PR-AUC={result['pr_auc']:.4f}" if "pr_auc" in result else ""))

    if write_json:
        with open(C.OUTPUTS / "metrics" / f"bert_{task_name}.json", "w") as f:
            json.dump({"bert": result}, f, indent=2)
    return result


def run_seed_sweep(task_name, cfg, seeds=(42, 43, 44), write_canonical=True):
    """Train the same config multiple times with different train_seed (model init + data
    shuffling + which training docs get subsampled), eval held fixed on C.SEED's test subsample.
    Reports mean/std instead of a single run, since picking one seed after seeing the spread
    would be post hoc selection. If write_canonical, also (re)writes the normal bert_{task}.json
    using the C.SEED trial (the project's standing convention, not chosen for its result) with
    the sweep's mean/std attached so evaluate.py can flag it as a smoke test if the spread is
    wide."""
    rows, full_results = [], {}
    for seed in seeds:
        result = run_task(task_name, cfg, train_seed=seed, test_seed=C.SEED,
                           write_json=False, write_predictions=False)
        full_results[seed] = result
        row = {"seed": seed, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}
        if "positive_f1" in result:
            row.update({
                "positive_f1": result["positive_f1"],
                "pr_auc": result["pr_auc"],
                "roc_auc": result["roc_auc"],
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c != "seed"]
    summary = {
        "task": task_name,
        "mode": cfg["mode"],
        "seeds": list(seeds),
        "test_seed": C.SEED,
        "runs": rows,
        "mean": {c: float(df[c].mean()) for c in metric_cols},
        "std": {c: float(df[c].std(ddof=0)) for c in metric_cols},
    }

    print(f"\n=== {task_name} seed sweep ({cfg['mode']}, test_seed={C.SEED}) ===")
    print(df.round(4).to_string(index=False))
    print("mean:", {k: round(v, 4) for k, v in summary["mean"].items()})
    print("std: ", {k: round(v, 4) for k, v in summary["std"].items()})

    sweep_path = C.OUTPUTS / "metrics" / f"bert_{task_name}_seed_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {sweep_path}")

    if write_canonical and C.SEED in full_results:
        canonical = dict(full_results[C.SEED])
        canonical["seed_sweep_seeds"] = list(seeds)
        for c in metric_cols:
            canonical[f"seed_sweep_mean_{c}"] = summary["mean"][c]
            canonical[f"seed_sweep_std_{c}"] = summary["std"][c]
        canonical_path = C.OUTPUTS / "metrics" / f"bert_{task_name}.json"
        with open(canonical_path, "w") as f:
            json.dump({"bert": canonical}, f, indent=2)
        print(f"wrote {canonical_path} (seed={C.SEED} trial, with sweep mean/std attached)")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--task", choices=["product", "escalation", "both"], default="both")
    parser.add_argument("--seed-sweep", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL

    (C.OUTPUTS / "metrics").mkdir(parents=True, exist_ok=True)

    if args.seed_sweep:
        tasks = ("product", "escalation") if args.task == "both" else (args.task,)
        for t in tasks:
            run_seed_sweep(t, cfg, seeds=args.seeds)
        return

    if args.task in ("product", "both"):
        run_task("product", cfg)
    if args.task in ("escalation", "both"):
        run_task("escalation", cfg)


if __name__ == "__main__":
    main()
