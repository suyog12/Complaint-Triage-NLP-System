import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from src import config as C
from src.preprocess import load_split

FIGURES = C.OUTPUTS / "figures"
METRICS = C.OUTPUTS / "metrics"

# Auto-route threshold is chosen on validation as the lowest confidence cutoff that hits this
# accuracy floor within the auto-routed subset. 95% is a business assumption (high-confidence
# bar for skipping human review), not derived from the data. The full sweep is saved alongside
# so a reader can pick a different floor.
ROUTING_ACCURACY_TARGET = 0.95
SWEEP_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

# A missed escalation sits in the standard queue past its regulatory deadline; a false alarm
# costs one analyst reading a routine complaint. 10x is a stated business input, not something
# the data told us.
MISS_TO_FALSE_ALARM_COST_RATIO = 10.0

# Same logic for routing: a misroute costs some multiple of a manual review, since a misroute
# means the complaint bounces between teams before anyone works it, versus a manual review
# costing one analyst's time to route it correctly by hand. No single ratio is given for
# routing the way 10:1 was given for escalation, so report the tradeoff at a few candidate
# ratios instead of assuming one.
ROUTING_COST_RATIOS = [2.0, 5.0, 10.0]


def _load_artifacts():
    product_vec = joblib.load(C.MODELS / "product_tfidf.joblib")
    product_model = joblib.load(C.MODELS / "product_logistic_regression.joblib")
    escalation_vec = joblib.load(C.MODELS / "escalation_tfidf.joblib")
    escalation_model = joblib.load(C.MODELS / "escalation_logistic_regression.joblib")
    return product_vec, product_model, escalation_vec, escalation_model


def _score(df, product_vec, product_model, escalation_vec, escalation_model):
    Xp = product_vec.transform(df.narrative)
    proba_p = product_model.predict_proba(Xp)
    routed_to = product_model.classes_[proba_p.argmax(axis=1)]
    routing_confidence = proba_p.max(axis=1)

    Xe = escalation_vec.transform(df.narrative)
    escalation_risk = escalation_model.predict_proba(Xe)[:, 1]

    out = df[["complaint_id", "product", "escalated"]].copy()
    out["routed_to"] = routed_to
    out["routing_confidence"] = routing_confidence
    out["routing_correct"] = out["routed_to"] == out["product"]
    out["escalation_risk"] = escalation_risk
    return out


def _pick_routing_threshold(val_scored, target_accuracy=ROUTING_ACCURACY_TARGET):
    for t in np.arange(0.0, 1.001, 0.01):
        subset = val_scored[val_scored.routing_confidence >= t]
        if len(subset) == 0:
            continue
        acc = subset.routing_correct.mean()
        if acc >= target_accuracy:
            return float(t), float(acc), len(subset) / len(val_scored)
    t = 1.0
    subset = val_scored[val_scored.routing_confidence >= t]
    acc = subset.routing_correct.mean() if len(subset) else float("nan")
    return t, acc, len(subset) / len(val_scored)


def _routing_sweep(scored, thresholds, applied_threshold, ratios=ROUTING_COST_RATIOS):
    all_t = sorted(set(thresholds) | {applied_threshold})
    n = len(scored)
    rows = []
    for t in all_t:
        subset = scored[scored.routing_confidence >= t]
        manual = n - len(subset)
        misroute = int((~subset.routing_correct).sum()) if len(subset) else 0
        row = {
            "threshold": t,
            "is_applied": t == applied_threshold,
            "auto_route_share": len(subset) / n,
            "auto_route_accuracy": subset.routing_correct.mean() if len(subset) else float("nan"),
            "misroute_count": misroute,
            "manual_review_count": manual,
        }
        for r in ratios:
            row[f"expected_cost_per_1000_ratio_{int(r)}"] = (misroute * r + manual) / n * 1000
        rows.append(row)
    return pd.DataFrame(rows)


def _cost_winners(df, ratios=ROUTING_COST_RATIOS):
    winners = {}
    for r in ratios:
        col = f"expected_cost_per_1000_ratio_{int(r)}"
        winners[r] = float(df.loc[df[col].idxmin(), "threshold"])
    return winners


def _implied_ratio_range(df, applied_threshold, r_min=0.1, r_max=30.0, r_step=0.1):
    """The range of misroute:review cost ratios for which the applied threshold is the
    cost-minimizer among the candidate thresholds in df. None if it never wins in [r_min, r_max]."""
    counts = {row.threshold: (row.misroute_count, row.manual_review_count) for row in df.itertuples()}
    hits = []
    r = r_min
    while r <= r_max + 1e-9:
        best_t = min(counts, key=lambda t: counts[t][0] * r + counts[t][1])
        if best_t == applied_threshold:
            hits.append(r)
        r += r_step
    return (min(hits), max(hits)) if hits else None


def _score_from_proba(y_true, proba, thresholds=SWEEP_THRESHOLDS, cost_ratio=MISS_TO_FALSE_ALARM_COST_RATIO):
    """Same cost-sweep logic as a LogReg priority sweep, but taking raw arrays instead of a
    scored dataframe, so it also works on BERT's saved probability arrays."""
    rows = []
    n = len(y_true)
    n_pos = int((y_true == 1).sum())
    for t in thresholds:
        pred_high = proba >= t
        tp = int((pred_high & (y_true == 1)).sum())
        fp = int((pred_high & (y_true == 0)).sum())
        fn = int((~pred_high & (y_true == 1)).sum())
        rows.append({
            "threshold": t,
            "high_share": pred_high.sum() / n,
            "recall": tp / n_pos if n_pos else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "false_negatives": fn,
            "false_positives": fp,
            "expected_cost_per_1000": (cost_ratio * fn + fp) / n * 1000,
        })
    return pd.DataFrame(rows)


def _priority_sweep(scored, thresholds=SWEEP_THRESHOLDS):
    return _score_from_proba(scored.escalated.to_numpy(), scored.escalation_risk.to_numpy(), thresholds)


def _bert_escalation_comparison(logreg_row):
    """BERT's own cost-optimal escalation threshold, tuned on validation and applied once to
    test - same discipline as the LogReg priority threshold, so the two models are compared at
    their respective tuned operating points instead of a tuned model against a threshold-0.5
    one. logreg_row carries LogReg's already-computed test-set stats at its own threshold, in
    the same schema, for a direct side-by-side table."""
    val_proba_path = C.OUTPUTS / "escalation_bert_val_proba.npy"
    test_proba_path = C.OUTPUTS / "escalation_bert_proba.npy"
    if not (val_proba_path.exists() and test_proba_path.exists()):
        return None, "not found - run src.transformer on the full split to generate them"

    val = load_split("val")
    test = load_split("test")
    val_proba = np.load(val_proba_path)
    test_proba = np.load(test_proba_path)
    if len(val_proba) != len(val) or len(test_proba) != len(test):
        # BERT was run on a subsample (e.g. quick mode) - not the full val/test splits, so its
        # threshold isn't comparable to LogReg's, which is tuned on the full validation split.
        return None, (
            f"found but wrong size (val={len(val_proba)}, expected {len(val)}; "
            f"test={len(test_proba)}, expected {len(test)}) - these look like a quick-mode "
            "subsample, not a full-split run. Re-run src.transformer without --quick"
        )

    val_sweep = _score_from_proba(val["escalated"].to_numpy(), val_proba)
    bert_threshold = float(val_sweep.loc[val_sweep.expected_cost_per_1000.idxmin(), "threshold"])

    y_true = test["escalated"].to_numpy()
    pred_high = test_proba >= bert_threshold
    tp = int((pred_high & (y_true == 1)).sum())
    fp = int((pred_high & (y_true == 0)).sum())
    fn = int((~pred_high & (y_true == 1)).sum())
    n = len(y_true)
    n_pos = int((y_true == 1).sum())

    bert_row = {
        "model": "BERT (full)",
        "threshold": bert_threshold,
        "high_share": float(pred_high.mean()),
        "recall": tp / n_pos if n_pos else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "false_negatives": fn,
        "false_positives": fp,
        "cost_per_1000": (MISS_TO_FALSE_ALARM_COST_RATIO * fn + fp) / n * 1000,
    }
    comparison = pd.DataFrame([dict(logreg_row, model="Logistic Regression"), bert_row])
    return {"val_sweep": val_sweep, "comparison": comparison}, None


def _plot_priority_sweep(df, chosen_threshold, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(df.threshold, df.recall, marker="o", label="Recall of true escalations")
    ax.plot(df.threshold, df.precision, marker="o", label="Precision")
    ax.plot(df.threshold, df.high_share, marker="o", label="HIGH share of volume")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", label=f"chosen ({chosen_threshold:.2f})")
    ax.set_xlabel("Priority threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Escalation priority: operating points")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(df.threshold, df.expected_cost_per_1000, marker="o", color="firebrick")
    ax2.axvline(chosen_threshold, color="gray", linestyle="--")
    ax2.set_xlabel("Priority threshold")
    ax2.set_ylabel("Expected cost per 1,000 complaints\n(miss = 10x false alarm)")
    ax2.set_title("Cost-minimizing threshold")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_routing_sweep(df, chosen_threshold, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df.threshold, df.auto_route_accuracy, marker="o", label="Auto-route accuracy")
    ax.plot(df.threshold, df.auto_route_share, marker="o", label="Auto-route share of volume")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", label=f"chosen ({chosen_threshold:.2f})")
    ax.set_xlabel("Routing confidence threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Product routing: confidence threshold tradeoff")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_routing_cost_sweep(df, chosen_threshold, ratios, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in ratios:
        col = f"expected_cost_per_1000_ratio_{int(r)}"
        ax.plot(df.threshold, df[col], marker="o", label=f"misroute:review = {int(r)}:1")
    ax.axvline(chosen_threshold, color="gray", linestyle="--", label=f"applied ({chosen_threshold:.2f})")
    ax.set_xlabel("Routing confidence threshold")
    ax.set_ylabel("Expected cost per 1,000 complaints")
    ax.set_title("Routing threshold: cost under candidate misroute:review ratios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    product_vec, product_model, escalation_vec, escalation_model = _load_artifacts()
    val = load_split("val")
    test = load_split("test")

    val_scored = _score(val, product_vec, product_model, escalation_vec, escalation_model)

    routing_threshold, val_route_acc, val_route_cov = _pick_routing_threshold(val_scored)
    routing_sweep = _routing_sweep(val_scored, SWEEP_THRESHOLDS, routing_threshold)
    routing_sweep.to_csv(METRICS / "routing_threshold_sweep.csv", index=False)
    _plot_routing_sweep(routing_sweep, routing_threshold, FIGURES / "routing_threshold_sweep.png")
    _plot_routing_cost_sweep(routing_sweep, routing_threshold, ROUTING_COST_RATIOS,
                              FIGURES / "routing_cost_sweep.png")
    routing_cost_winners = _cost_winners(routing_sweep)
    routing_implied_range = _implied_ratio_range(routing_sweep, routing_threshold)

    priority_sweep = _priority_sweep(val_scored)
    priority_sweep.to_csv(METRICS / "priority_threshold_sweep.csv", index=False)
    priority_threshold = float(priority_sweep.loc[priority_sweep.expected_cost_per_1000.idxmin(), "threshold"])
    _plot_priority_sweep(priority_sweep, priority_threshold, FIGURES / "priority_threshold_sweep.png")

    print("routing threshold sweep (validation):")
    print(routing_sweep.round(4).to_string(index=False))
    print(f"\napplied routing threshold: {routing_threshold:.2f} "
          f"(lowest threshold clearing {ROUTING_ACCURACY_TARGET:.0%} accuracy floor; "
          f"val accuracy={val_route_acc:.4f}, coverage={val_route_cov:.4f})")
    print("cost-minimizing threshold by misroute:review ratio: " +
          ", ".join(f"{int(r)}:1 -> {t:.2f}" for r, t in routing_cost_winners.items()))
    if routing_implied_range is not None:
        lo, hi = routing_implied_range
        print(f"applied threshold {routing_threshold:.2f} is cost-optimal for misroute:review "
              f"ratios in roughly [{lo:.1f}:1, {hi:.1f}:1]")
    else:
        print(f"applied threshold {routing_threshold:.2f} is not cost-optimal at any ratio "
              f"in [0.1:1, 30:1] among the candidate thresholds; it is a pure accuracy-floor "
              f"choice, not a cost-minimizing one")

    print("\npriority threshold sweep (validation, cost = 10x miss + 1x false alarm):")
    print(priority_sweep.round(4).to_string(index=False))
    print(f"\napplied priority threshold: {priority_threshold:.2f} (minimizes expected cost per 1,000 complaints)")

    test_scored = _score(test, product_vec, product_model, escalation_vec, escalation_model)
    test_scored["action"] = np.where(
        test_scored.routing_confidence >= routing_threshold, "AUTO ROUTE", "MANUAL REVIEW"
    )
    test_scored["priority"] = np.where(
        test_scored.escalation_risk >= priority_threshold, "HIGH", "STANDARD"
    )

    n = len(test_scored)
    auto = test_scored[test_scored.action == "AUTO ROUTE"]
    manual = test_scored[test_scored.action == "MANUAL REVIEW"]
    high = test_scored[test_scored.priority == "HIGH"]
    true_escalated = test_scored.escalated == 1
    caught = ((test_scored.priority == "HIGH") & true_escalated).sum()
    high_recall = caught / true_escalated.sum() if true_escalated.sum() else float("nan")
    high_precision = precision_score(
        test_scored.escalated, (test_scored.priority == "HIGH").astype(int),
        pos_label=1, zero_division=0,
    )
    logreg_fn = int((true_escalated & (test_scored.priority == "STANDARD")).sum())
    logreg_fp = int((~true_escalated & (test_scored.priority == "HIGH")).sum())

    logreg_priority_row = {
        "threshold": priority_threshold,
        "high_share": len(high) / n,
        "recall": float(high_recall),
        "precision": float(high_precision),
        "false_negatives": logreg_fn,
        "false_positives": logreg_fp,
        "cost_per_1000": (MISS_TO_FALSE_ALARM_COST_RATIO * logreg_fn + logreg_fp) / n * 1000,
    }
    bert_comparison, bert_skip_reason = _bert_escalation_comparison(logreg_priority_row)
    if bert_comparison is None:
        print(f"\nBERT escalation probabilities {bert_skip_reason}. "
              "Skipping BERT vs LogReg comparison.")
    else:
        bert_comparison["val_sweep"].to_csv(METRICS / "bert_priority_threshold_sweep.csv", index=False)
        print("\nBERT escalation priority sweep (validation, cost = 10x miss + 1x false alarm):")
        print(bert_comparison["val_sweep"].round(4).to_string(index=False))
        print("\nLogReg vs BERT at their respective cost-optimal thresholds (tuned on "
              "validation, applied once to test):")
        print(bert_comparison["comparison"].round(4).to_string(index=False))
        bert_comparison["comparison"].to_csv(METRICS / "bert_vs_logreg_priority.csv", index=False)

    summary = {
        "routing_confidence_threshold": routing_threshold,
        "routing_threshold_selection": (
            f"lowest threshold clearing {ROUTING_ACCURACY_TARGET:.0%} val accuracy floor "
            "(business assumption; see routing_threshold_sweep.csv for alternatives)"
        ),
        "routing_cost_winners_by_misroute_review_ratio": {
            f"{int(r)}:1": t for r, t in routing_cost_winners.items()
        },
        "routing_applied_threshold_implied_cost_ratio_range": routing_implied_range,
        "priority_escalation_threshold": priority_threshold,
        "priority_threshold_selection": (
            "minimizes expected cost per 1,000 complaints, assuming a missed escalation costs "
            "10x a false alarm (business assumption; see priority_threshold_sweep.csv for alternatives)"
        ),
        "miss_to_false_alarm_cost_ratio": MISS_TO_FALSE_ALARM_COST_RATIO,
        "val_auto_route_accuracy": val_route_acc,
        "val_auto_route_coverage": val_route_cov,
        "test_docs": n,
        "test_auto_route_share": len(auto) / n,
        "test_auto_route_accuracy": auto.routing_correct.mean() if len(auto) else float("nan"),
        "test_manual_review_share": len(manual) / n,
        "test_manual_review_accuracy": manual.routing_correct.mean() if len(manual) else float("nan"),
        "test_high_priority_share": len(high) / n,
        "test_high_priority_recall_of_true_escalations": float(high_recall),
        "test_high_priority_precision": float(high_precision),
    }
    if bert_comparison is not None:
        bert_row = bert_comparison["comparison"].set_index("model").loc["BERT (full)"]
        summary["bert_escalation_threshold"] = float(bert_row["threshold"])
        summary["bert_vs_logreg_cost_per_1000"] = {
            "logistic_regression": logreg_priority_row["cost_per_1000"],
            "bert_full": float(bert_row["cost_per_1000"]),
        }
    print("\n" + json.dumps(summary, indent=2))

    test_scored.to_csv(C.OUTPUTS / "triage_simulation.csv", index=False)
    with open(METRICS / "triage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run()
