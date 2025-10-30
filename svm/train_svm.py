# print("hello svm")
# -*- coding: utf-8 -*-
"""
Train SVM on full MNIST using image files indexed by TSVs.
- Cross-validate kernels (linear, rbf) and hyperparameters.
- Optional PCA for speed.
- Evaluate on held-out test set.
- Save best model and CV table.

Usage:
    python svm/train_svm.py --root . --pca 0.95
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_from_tsv(tsv_path: Path):
    """
    Read TSV: <relative_image_path>\t<label>
    Return X (N, 784) float32 in [0,1], y (N,) int64
    """
    base = tsv_path.parent
    xs, ys = [], []
    with tsv_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            rel_path, label = row[0], row[1]
            img_path = (base / rel_path).resolve()
            # load grayscale 28x28
            im = Image.open(img_path).convert("L")
            # ensure 28x28 just in case
            if im.size != (28, 28):
                im = im.resize((28, 28))
            arr = np.asarray(im, dtype=np.float32) / 255.0  # [0,1]
            xs.append(arr.flatten())
            ys.append(int(label))
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".",
                        help="Project root containing gt-train.tsv and gt-test.tsv")
    parser.add_argument("--pca", type=float, default=None,
                        help="If set (e.g., 0.95), apply PCA keeping given explained variance")
    parser.add_argument("--cv", type=int, default=3, help="CV folds")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tsv_train = root / "gt-train.tsv"
    tsv_test = root / "gt-test.tsv"
    if not tsv_train.exists() or not tsv_test.exists():
        raise FileNotFoundError(
            f"Could not find gt-train.tsv/gt-test.tsv under {root}. "
            "Please set --root to the directory that contains the TSV files."
        )

    print(f"[INFO] Loading training set from {tsv_train} ...")
    X_train, y_train = load_from_tsv(tsv_train)
    print(f"[INFO] Training samples: {X_train.shape}, classes: {len(np.unique(y_train))}")

    print(f"[INFO] Loading test set from {tsv_test} ...")
    X_test, y_test = load_from_tsv(tsv_test)
    print(f"[INFO] Test samples: {X_test.shape}")

    steps = [("scaler", StandardScaler())]
    if args.pca is not None:
        steps.append(("pca", PCA(n_components=args.pca, svd_solver="full", whiten=False)))
        print(f"[INFO] PCA enabled: keep {args.pca} explained variance")

    steps.append(("svc", SVC()))
    pipe = Pipeline(steps)

    # Parameter grid for both linear and rbf
    param_grid = [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.5, 1, 2, 5, 10],
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [1, 2, 5, 10],
            "svc__gamma": ["scale", 0.01, 0.005, 0.001],
        },
    ]

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    print("[INFO] Starting grid search ...")
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=args.n_jobs,
        verbose=2,
        return_train_score=False,
    )
    gscv.fit(X_train, y_train)

    print("\n[RESULT] Best CV score: {:.4f}".format(gscv.best_score_))
    print("[RESULT] Best params:", gscv.best_params_)

    # Evaluate on test set with best estimator
    best = gscv.best_estimator_
    y_pred = best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("\n[RESULT] Test accuracy: {:.4f}".format(test_acc))
    print("\n[REPORT]\n", classification_report(y_test, y_pred, digits=4))

    # Prepare output paths
    out_dir = Path("svm")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = out_dir / "best_svm.joblib"
    joblib.dump(best, model_path)
    print(f"[SAVE] Best model saved to: {model_path}")

    # Save CV table
    import pandas as pd  # lazy import to keep top clean
    cv_df = pd.DataFrame(gscv.cv_results_)
    cv_path = out_dir / "svm_cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"[SAVE] GridSearchCV results saved to: {cv_path}")

    # Also save a tiny summary for your report
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Best CV accuracy: {:.4f}\n".format(gscv.best_score_))
        f.write("Best params: {}\n".format(gscv.best_params_))
        f.write("Test accuracy: {:.4f}\n".format(test_acc))
    print(f"[SAVE] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
