import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score
from src import config as C
from src.preprocess import load_split
from src.triage import _load_artifacts as _load_triage_artifacts
from src.triage import _score as _triage_score

HYBRID_THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.1, 0.96, 0.05)]

FIGURES = C.OUTPUTS / "figures"
METRICS = C.OUTPUTS / "metrics"

# True population counts by product, from the full 3.82M-row filtered CFPB set
POPULATION = {
    "Credit reporting": 2_499_358,
    "Debt collection": 442_907,
    "Credit card": 245_757,
    "Bank account": 200_206,
    "Mortgage": 145_840,
    "Money transfer / virtual currency": 121_554,
    "Student loan": 62_351,
    "Vehicle or consumer loan": 62_062,
    "Payday / title / personal loan": 36_803,
}

MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "bert": "BERT",
    "majority_baseline": "Majority baseline",
}


def _load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _merged_models(baseline_file, bert_file):
    models = dict(_load_json(METRICS / baseline_file) or {})
    bert = _load_json(METRICS / bert_file)
    if bert and "bert" in bert:
        models["bert"] = bert["bert"]
    return models


def _model_label(name, m):
    """BERT rows are only meaningful with their training size and epoch count attached -
    otherwise a quick-mode number (n=2,000) reads as if it were comparable to a baseline
    trained on the full 37,800-doc split, which it is not."""
    if name != "bert":
        return MODEL_LABELS.get(name, name)
    mode = m.get("mode", "?")
    n = m.get("train_n_docs", "?")
    epochs = m.get("epochs", "?")
    epoch_word = "epoch" if epochs == 1 else "epochs"
    return f"BERT ({mode}, n={n}, {epochs} {epoch_word})"


def _quick_mode_caveat(models):
    if not any(m.get("mode") == "quick" for m in models.values()):
        return None
    return (
        "Note: the BERT row above is from a quick-mode CPU smoke test (DistilBERT, n=2,000, "
        "1 epoch), not the full run (bert-base-uncased, n=37,800, 2 epochs, GPU). Quick and "
        "full runs use different training sizes and are not directly comparable to each other "
        "or to the sklearn baselines."
    )


def _seed_sweep_caveat(models):
    m = models.get("bert", {})
    seeds = m.get("seed_sweep_seeds")
    if not seeds:
        return None
    lines = [f"Seed sweep ({len(seeds)} seeds: {seeds}, test set held fixed):"]
    for key in ("positive_f1", "pr_auc", "accuracy", "macro_f1"):
        mean_key, std_key = f"seed_sweep_mean_{key}", f"seed_sweep_std_{key}"
        if mean_key in m:
            lines.append(f"  {key}: mean={m[mean_key]:.4f}  std={m[std_key]:.4f}")
    lines.append(
        "This spread means the single-run BERT quick-mode number above is not a reliable "
        "performance estimate. Report quick mode as a smoke test that verifies the training "
        "code path runs correctly, not as a performance comparison against the sklearn "
        "baselines. See outputs/metrics/bert_escalation_seed_sweep.json for the per-seed runs."
    )
    return "\n".join(lines)


def _timing(task, name, text):
    """Re-measure inference end to end (raw text -> vectorize -> predict), not just predict()
    on an already-vectorized matrix. Needed so the sklearn numbers are comparable to BERT's,
    which will include tokenization. Returns None if no joblib artifacts exist for (task, name)
    — e.g. for a future bert entry, whose own timing fields get used as a fallback instead."""
    vec_path = C.MODELS / f"{task}_tfidf.joblib"
    model_path = C.MODELS / f"{task}_{name}.joblib"
    if not (vec_path.exists() and model_path.exists()):
        return None
    vec = joblib.load(vec_path)
    model = joblib.load(model_path)
    t0 = time.perf_counter()
    X = vec.transform(text)
    t1 = time.perf_counter()
    model.predict(X)
    t2 = time.perf_counter()
    return {"predict_only_seconds": t2 - t1, "end_to_end_seconds": t2 - t0}


def _timing_row(task, name, m, text, n_docs):
    timing = _timing(task, name, text)
    if timing is None:
        # No joblib artifacts for this model (e.g. bert) - use whatever it self-reported.
        # Its eval set may not be the full sklearn test split (quick mode uses 1,000 docs),
        # so use its own doc count rather than the sklearn n_docs passed in.
        fallback = m.get("inference_seconds_8100_docs")
        timing = {
            "end_to_end_seconds": m.get("end_to_end_seconds", fallback),
            "predict_only_seconds": m.get("predict_seconds", fallback),
        }
        n_docs = m.get("inference_n_docs", n_docs)
    end_to_end = timing["end_to_end_seconds"]
    predict_only = timing["predict_only_seconds"]
    return {
        "end_to_end_inference_seconds": end_to_end,
        "predict_only_inference_seconds": predict_only,
        "end_to_end_docs_per_second": n_docs / end_to_end if end_to_end else None,
        "predict_only_docs_per_second": n_docs / predict_only if predict_only else None,
    }


def _significance_test(y_true, pred_a, pred_b, n_boot=2000, seed=None, label_a="hybrid", label_b="bert"):
    """Is pred_a genuinely more accurate than pred_b, or within noise? McNemar's exact test on
    paired correct/incorrect outcomes (standard for comparing two classifiers on the same test
    set), plus a paired bootstrap on the macro-F1 difference itself, since McNemar only speaks
    to accuracy and the headline claim is about macro-F1."""
    seed = C.SEED if seed is None else seed
    y_true = np.asarray(y_true)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_right_b_wrong = int((correct_a & ~correct_b).sum())
    b_right_a_wrong = int((~correct_a & correct_b).sum())
    n_discordant = a_right_b_wrong + b_right_a_wrong
    mcnemar_p = (
        float(binomtest(a_right_b_wrong, n_discordant, p=0.5).pvalue)
        if n_discordant else float("nan")
    )

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (
            f1_score(y_true[idx], pred_a[idx], average="macro")
            - f1_score(y_true[idx], pred_b[idx], average="macro")
        )
    ci_low, ci_high = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    boot_p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_discordant": n_discordant,
        f"{label_a}_right_{label_b}_wrong": a_right_b_wrong,
        f"{label_b}_right_{label_a}_wrong": b_right_a_wrong,
        "mcnemar_exact_p_value": mcnemar_p,
        "bootstrap_n_resamples": n_boot,
        "bootstrap_macro_f1_diff_mean": float(diffs.mean()),
        "bootstrap_macro_f1_diff_ci95": [ci_low, ci_high],
        "bootstrap_two_sided_p_value": boot_p,
    }


def _hybrid_threshold_sweep(thresholds=HYBRID_THRESHOLD_GRID):
    """The handoff threshold (when to trust LogReg over BERT) is tuned independently here, not
    reused from triage.py's 0.83 auto-route threshold - that one was tuned to hit a 95% accuracy
    floor for skipping human review entirely, a different objective from maximizing the hybrid's
    combined macro-F1. Swept on validation, applied once to test, same discipline as every other
    threshold in this project. Returns None if BERT's per-document val/test predictions aren't
    available (missing, or a quick-mode subsample not covering the full splits)."""
    val_bert_path = C.OUTPUTS / "product_bert_val_pred.npy"
    test_bert_path = C.OUTPUTS / "product_bert_pred.npy"
    if not (val_bert_path.exists() and test_bert_path.exists()):
        return None

    product_vec, product_model, escalation_vec, escalation_model = _load_triage_artifacts()
    val = load_split("val")
    test = load_split("test")
    val_scored = _triage_score(val, product_vec, product_model, escalation_vec, escalation_model)
    test_scored = _triage_score(test, product_vec, product_model, escalation_vec, escalation_model)

    bert_val_pred = np.load(val_bert_path, allow_pickle=True)
    bert_test_pred = np.load(test_bert_path, allow_pickle=True)
    if len(bert_val_pred) != len(val_scored) or len(bert_test_pred) != len(test_scored):
        return None

    bert_m = _load_json(METRICS / "bert_product.json")
    if not bert_m or "bert" not in bert_m:
        return None
    bert_m = bert_m["bert"]

    logreg_timing = _timing("product", "logistic_regression", test.narrative)
    logreg_dps = len(test_scored) / logreg_timing["end_to_end_seconds"] if logreg_timing else None
    bert_dps = bert_m["inference_n_docs"] / bert_m["end_to_end_seconds"]

    def blended_dps(bert_share):
        if logreg_dps is None:
            return None
        return 1.0 / ((1 - bert_share) / logreg_dps + bert_share / bert_dps)

    y_val = val_scored["product"].to_numpy()
    val_conf = val_scored.routing_confidence.to_numpy()
    val_logreg_pred = val_scored.routed_to.to_numpy()

    rows = []
    for t in thresholds:
        use_logreg = val_conf >= t
        hybrid_pred = np.where(use_logreg, val_logreg_pred, bert_val_pred)
        bert_share = float((~use_logreg).mean())
        rows.append({
            "threshold": float(t),
            "val_macro_f1": f1_score(y_val, hybrid_pred, average="macro"),
            "bert_share": bert_share,
            "blended_docs_per_second": blended_dps(bert_share),
        })
    sweep_df = pd.DataFrame(rows)
    chosen_threshold = float(sweep_df.loc[sweep_df.val_macro_f1.idxmax(), "threshold"])

    y_test = test_scored["product"].to_numpy()
    test_conf = test_scored.routing_confidence.to_numpy()
    test_logreg_pred = test_scored.routed_to.to_numpy()
    use_logreg_test = test_conf >= chosen_threshold
    hybrid_test_pred = np.where(use_logreg_test, test_logreg_pred, bert_test_pred)
    bert_share_test = float((~use_logreg_test).mean())

    significance = _significance_test(y_test, hybrid_test_pred, bert_test_pred, seed=C.SEED)

    return {
        "sweep": sweep_df,
        "chosen_threshold": chosen_threshold,
        "test_accuracy": float((hybrid_test_pred == y_test).mean()),
        "test_macro_f1": float(f1_score(y_test, hybrid_test_pred, average="macro")),
        "test_bert_share": bert_share_test,
        "test_logreg_share": 1.0 - bert_share_test,
        "test_blended_docs_per_second": blended_dps(bert_share_test),
        "bert_test_accuracy": float((bert_test_pred == y_test).mean()),
        "bert_test_macro_f1": float(f1_score(y_test, bert_test_pred, average="macro")),
        "significance_vs_bert": significance,
    }


def _plot_hybrid_sweep(df, chosen_threshold, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(df.threshold, df.val_macro_f1, marker="o", color="darkgreen")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", label=f"chosen ({chosen_threshold:.2f})")
    ax.set_xlabel("Handoff threshold (LogReg confidence)")
    ax.set_ylabel("Validation macro F1 (hybrid)")
    ax.set_title("Hybrid handoff threshold: macro-F1")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(df.threshold, df.blended_docs_per_second, marker="o", color="steelblue")
    ax2.axvline(chosen_threshold, color="gray", linestyle="--")
    ax2.set_xlabel("Handoff threshold (LogReg confidence)")
    ax2.set_ylabel("Blended docs/sec (validation share)")
    ax2.set_title("Hybrid handoff threshold: throughput")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hybrid_product_row(hybrid_result):
    return {
        "model": f"Hybrid (LogReg conf>={hybrid_result['chosen_threshold']:.2f}, else BERT)",
        "accuracy": hybrid_result["test_accuracy"],
        "macro_f1": hybrid_result["test_macro_f1"],
        "train_seconds": None,
        "end_to_end_inference_seconds": None,
        "predict_only_inference_seconds": None,
        "end_to_end_docs_per_second": hybrid_result["test_blended_docs_per_second"],
        "predict_only_docs_per_second": None,
        "mode": "hybrid",
        "logreg_share": hybrid_result["test_logreg_share"],
        "bert_share": hybrid_result["test_bert_share"],
    }


def product_comparison(hybrid_result=None):
    models = _merged_models("baselines_product.json", "bert_product.json")
    text = load_split("test").narrative
    n_docs = len(text)
    rows = []
    for name, m in models.items():
        row = {
            "model": _model_label(name, m),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "train_seconds": m["train_seconds"],
        }
        row.update(_timing_row("product", name, m, text, n_docs))
        row["mode"] = m.get("mode", "")
        rows.append(row)
    if hybrid_result is not None:
        rows.append(_hybrid_product_row(hybrid_result))
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "product_comparison.csv", index=False)
    return df


def escalation_comparison():
    models = _merged_models("baselines_escalation.json", "bert_escalation.json")
    text = load_split("test").narrative
    n_docs = len(text)
    rows = []
    for name, m in models.items():
        row = {
            "model": _model_label(name, m),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "positive_f1": m.get("positive_f1"),
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
        }
        if name != "majority_baseline":
            row.update(_timing_row("escalation", name, m, text, n_docs))
        row["mode"] = m.get("mode", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "escalation_comparison.csv", index=False)
    return df


def prevalence_weighted_accuracy():
    models = _merged_models("baselines_product.json", "bert_product.json")
    total_pop = sum(POPULATION.values())
    rows = []
    for name, m in models.items():
        report = m.get("report", {})
        weighted = sum(
            report[cls]["recall"] * pop
            for cls, pop in POPULATION.items()
            if cls in report
        ) / total_pop
        rows.append({
            "model": _model_label(name, m),
            "balanced_test_accuracy": m["accuracy"],
            "prevalence_weighted_accuracy": weighted,
            "gap": m["accuracy"] - weighted,
        })
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "prevalence_weighted_accuracy.csv", index=False)
    return df


def confusion_matrix_figure():
    pred_path = C.OUTPUTS / "product_logistic_regression_pred.npy"
    if not pred_path.exists():
        return
    te = load_split("test")
    y_true = te["product"].to_numpy()
    y_pred = np.load(pred_path, allow_pickle=True)
    labels = sorted(POPULATION.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Logistic Regression product routing\nrow-normalised confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="row-normalised share")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_logreg_product.png", dpi=150)
    plt.close(fig)


def learning_curve_figure():
    multi = _load_json(METRICS / "learning_curve_multiseed.json")
    single = _load_json(METRICS / "learning_curve.json")
    if multi is None and single is None:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    if multi is not None:
        n = [row["n"] for row in multi]
        mean = [row["mean_macro_f1"] for row in multi]
        std = [row["std_macro_f1"] for row in multi]
        ax.errorbar(n, mean, yerr=std, marker="o", capsize=4,
                    label="Logistic Regression (mean ± std, seeds 42/43/44)")
    else:
        n = [row["n"] for row in single]
        f1 = [row["macro_f1"] for row in single]
        ax.plot(n, f1, marker="o", label="Logistic Regression")
    ax.set_xlabel("Training documents")
    ax.set_ylabel("Macro F1 (product routing, test split)")
    ax.set_title("Learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curve.png", dpi=150)
    plt.close(fig)


def run():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    hybrid_result = _hybrid_threshold_sweep()
    if hybrid_result is None:
        print("Hybrid LogReg+BERT sweep skipped - outputs/product_bert_val_pred.npy and/or "
              "product_bert_pred.npy not found or not aligned to the full val/test splits. "
              "Run src.transformer on the full split (not --quick) to generate them.")
    else:
        hybrid_result["sweep"].to_csv(METRICS / "hybrid_threshold_sweep.csv", index=False)
        _plot_hybrid_sweep(hybrid_result["sweep"], hybrid_result["chosen_threshold"],
                            FIGURES / "hybrid_threshold_sweep.png")
        print("\n=== Hybrid handoff threshold sweep (validation) ===")
        print(hybrid_result["sweep"].round(4).to_string(index=False))
        print(f"\nchosen handoff threshold: {hybrid_result['chosen_threshold']:.2f} "
              "(maximizes hybrid validation macro-F1)")
        print("note: this is tuned independently of triage.py's 0.83 auto-route threshold, "
              "which optimizes a different objective (a 95% accuracy floor for skipping human "
              "review, not maximizing combined macro-F1) - "
              + ("they happen to coincide." if hybrid_result["chosen_threshold"] == 0.83
                 else f"they differ (0.83 vs {hybrid_result['chosen_threshold']:.2f})."))

        sig = hybrid_result["significance_vs_bert"]
        print(f"\nhybrid vs pure BERT on test (n_discordant={sig['n_discordant']}, "
              f"hybrid_right_bert_wrong={sig['hybrid_right_bert_wrong']}, "
              f"bert_right_hybrid_wrong={sig['bert_right_hybrid_wrong']}):")
        print(f"  McNemar exact p-value: {sig['mcnemar_exact_p_value']:.4f}")
        print(f"  bootstrap macro-F1 diff: {sig['bootstrap_macro_f1_diff_mean']:.4f} "
              f"(95% CI [{sig['bootstrap_macro_f1_diff_ci95'][0]:.4f}, "
              f"{sig['bootstrap_macro_f1_diff_ci95'][1]:.4f}], "
              f"n={sig['bootstrap_n_resamples']} resamples), "
              f"two-sided p={sig['bootstrap_two_sided_p_value']:.4f}")
        with open(METRICS / "hybrid_vs_bert_significance.json", "w") as f:
            json.dump({k: v for k, v in hybrid_result.items() if k != "sweep"}, f, indent=2)

    prod_df = product_comparison(hybrid_result)
    print("\n=== Product routing comparison ===")
    print(prod_df.round(4).to_string(index=False))
    prod_caveat = _quick_mode_caveat(_merged_models("baselines_product.json", "bert_product.json"))
    if prod_caveat:
        print(prod_caveat)

    esc_models = _merged_models("baselines_escalation.json", "bert_escalation.json")
    esc_df = escalation_comparison()
    print("\n=== Escalation risk comparison ===")
    print(esc_df.round(4).to_string(index=False))
    esc_caveat = _quick_mode_caveat(esc_models)
    if esc_caveat:
        print(esc_caveat)
    seed_caveat = _seed_sweep_caveat(esc_models)
    if seed_caveat:
        print(seed_caveat)

    prev_df = prevalence_weighted_accuracy()
    print("\n=== Prevalence-weighted accuracy (product routing) ===")
    print(prev_df.round(4).to_string(index=False))

    confusion_matrix_figure()
    learning_curve_figure()
    print(f"\nfigures written to {FIGURES}")
    print(f"tables written to {METRICS}")


if __name__ == "__main__":
    run()
