import json
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    average_precision_score, roc_auc_score,
)
from src import config as C
from src.preprocess import load_split


def vectorize(train_text, other_texts, **kw):
    vec = TfidfVectorizer(
        max_features=100_000, ngram_range=(1, 2),
        min_df=3, sublinear_tf=True, strip_accents="unicode", **kw
    )
    Xtr = vec.fit_transform(train_text)
    return vec, Xtr, [vec.transform(t) for t in other_texts]


def run_task(label_col, task_name):
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    vec, Xtr, (Xva, Xte) = vectorize(tr.narrative, [va.narrative, te.narrative])
    ytr, yva, yte = tr[label_col], va[label_col], te[label_col]
    print(f"\n=== {task_name} | features: {Xtr.shape[1]} ===")

    results = {}
    models = {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(
            max_iter=2000, C=4.0,
            class_weight="balanced" if label_col == "escalated" else None,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        train_s = time.time() - t0

        t0 = time.time()
        pred = model.predict(Xte)
        infer_s = time.time() - t0

        r = {
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro"),
            "train_seconds": round(train_s, 2),
            "inference_seconds_8100_docs": round(infer_s, 3),
            "report": classification_report(yte, pred, output_dict=True, zero_division=0),
        }
        if label_col == "escalated":
            proba = model.predict_proba(Xte)[:, 1]
            r["pr_auc"] = average_precision_score(yte, proba)
            r["roc_auc"] = roc_auc_score(yte, proba)
            r["positive_f1"] = f1_score(yte, pred, pos_label=1)
            np.save(C.OUTPUTS / f"{task_name}_{name}_proba.npy", proba)

        results[name] = r
        np.save(C.OUTPUTS / f"{task_name}_{name}_pred.npy", pred)
        joblib.dump(model, C.MODELS / f"{task_name}_{name}.joblib")
        print(f"{name:22s} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  "
              f"fit={train_s:.1f}s" + (f"  PR-AUC={r['pr_auc']:.4f}" if "pr_auc" in r else ""))

    if label_col == "escalated":
        majority = np.zeros(len(yte), dtype=int)
        results["majority_baseline"] = {
            "accuracy": accuracy_score(yte, majority),
            "macro_f1": f1_score(yte, majority, average="macro", zero_division=0),
        }
        print(f"{'majority_baseline':22s} acc={results['majority_baseline']['accuracy']:.4f}")

    joblib.dump(vec, C.MODELS / f"{task_name}_tfidf.joblib")
    with open(C.OUTPUTS / "metrics" / f"baselines_{task_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    C.MODELS.mkdir(exist_ok=True)
    (C.OUTPUTS / "metrics").mkdir(parents=True, exist_ok=True)
    run_task("product", "product")
    run_task("escalated", "escalation")