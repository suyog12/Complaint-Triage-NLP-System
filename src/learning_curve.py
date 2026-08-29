import argparse
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from src import config as C
from src.baselines import vectorize
from src.preprocess import load_split

SIZES = [1000, 2500, 5000, 10000, 20000, 37800]
SEEDS = [42, 43, 44]


def _score(tr, te, n, seed):
    sub = tr.sample(frac=1, random_state=seed).reset_index(drop=True).head(n)
    vec, Xtr, (Xte,) = vectorize(sub.narrative, [te.narrative])
    m = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, sub["product"])
    return f1_score(te["product"], m.predict(Xte), average="macro")


def run():
    tr, te = load_split("train"), load_split("test")
    tr = tr.sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    out = []
    for n in SIZES:
        sub = tr.head(n)
        vec, Xtr, (Xte,) = vectorize(sub.narrative, [te.narrative])
        m = LogisticRegression(max_iter=2000, C=4.0).fit(Xtr, sub["product"])
        f1 = f1_score(te["product"], m.predict(Xte), average="macro")
        out.append({"n": n, "macro_f1": f1})
        print(f"n={n:6d}  macroF1={f1:.4f}")
    with open(C.OUTPUTS / "metrics" / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2)


def run_multiseed():
    tr, te = load_split("train"), load_split("test")
    out = []
    for n in SIZES:
        scores = []
        for seed in SEEDS:
            f1 = _score(tr, te, n, seed)
            scores.append(f1)
            print(f"n={n:6d}  seed={seed}  macroF1={f1:.4f}")
        mean, std = float(np.mean(scores)), float(np.std(scores))
        out.append({"n": n, "seeds": SEEDS, "scores": scores, "mean_macro_f1": mean, "std_macro_f1": std})
        print(f"n={n:6d}  mean={mean:.4f}  std={std:.4f}")
    with open(C.OUTPUTS / "metrics" / "learning_curve_multiseed.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed", action="store_true")
    args = parser.parse_args()
    run_multiseed() if args.multiseed else run()