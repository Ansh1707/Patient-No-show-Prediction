import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Paths are relative to this script; --data can point to a CSV elsewhere.
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "KaggleV2-May-2016.csv"
OUT_DIR = BASE_DIR / "model"
SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # Fraction of the complete dataset.
N_FOLDS = 5
TARGET_RECALL = 0.75

CAT_COLS = ["Gender"]
NUM_COLS = ["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
            "Handcap", "SMS_received", "wait_days"]
FEATURES = CAT_COLS + NUM_COLS


def load_data(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["ScheduledDay", "AppointmentDay"])
    df["wait_days"] = (df["AppointmentDay"].dt.normalize()
                       - df["ScheduledDay"].dt.normalize()).dt.days
    df = df[df.Age >= 0]
    return df[FEATURES], (df["No-show"] == "Yes").astype(int)


def split_data(X, y):
    # Hold out test rows before any model or threshold selection.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, stratify=y_dev,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE), random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models(y_train):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preproc = ColumnTransformer([
        ("ohe", ohe, CAT_COLS),
        ("scale", StandardScaler(), NUM_COLS)
    ])

    def make_training_pipe(clf):
        return ImbPipeline([
            ("prep", clone(preproc)),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", clf)
        ])

    # Class weights must not depend on validation or test labels.
    neg, pos = np.bincount(y_train, minlength=2)
    scale_pos = neg / pos
    return {
        "LogReg": make_training_pipe(
            LogisticRegression(class_weight="balanced", solver="liblinear", random_state=SEED)),
        "RF": make_training_pipe(
            RandomForestClassifier(class_weight="balanced_subsample", n_estimators=300,
                                   n_jobs=-1, random_state=SEED)),
        "XGB": make_training_pipe(
            xgb.XGBClassifier(
                eval_metric="logloss", scale_pos_weight=scale_pos,
                n_estimators=400, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, verbosity=0, random_state=SEED)),
        "LGBM": make_training_pipe(
            lgb.LGBMClassifier(is_unbalance=True, n_estimators=400, learning_rate=0.1,
                               num_leaves=31, n_jobs=-1, verbose=-1, random_state=SEED))
    }


def select_threshold(y_val, proba_val, target_recall=TARGET_RECALL):
    _, recall, thresholds = precision_recall_curve(y_val, proba_val)
    # The last recall entry has no corresponding threshold.
    candidates = np.flatnonzero(recall[:-1] >= target_recall)
    if not len(candidates):
        raise ValueError("No validation threshold satisfies the target recall")
    return float(thresholds[candidates[-1]])


def train_model(csv_file=CSV_FILE, out_dir=OUT_DIR):
    X, y = load_data(csv_file)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Split sizes: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")

    models = build_models(y_train)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring={"pr_auc": "average_precision"}, n_jobs=-1)
        cv_scores[name] = float(res["test_pr_auc"].mean())
    best_name = max(cv_scores, key=cv_scores.get)
    best_train_pipe = models[best_name]
    print(f"Best model by training CV PR-AUC: {best_name} ({cv_scores[best_name]:.3f})")
    best_train_pipe.fit(X_train, y_train)

    # Freeze the threshold using validation data only.
    proba_val = best_train_pipe.predict_proba(X_val)[:, 1]
    tuned_threshold = select_threshold(y_val, proba_val)
    print(f"Validation-selected threshold for recall ≥ {TARGET_RECALL}: {tuned_threshold:.3f}")

    # Keep the fitted components: no refit after tuning, and no SMOTE at inference.
    prod_pipeline = SkPipeline([
        ("prep", best_train_pipe.named_steps["prep"]),
        ("clf", best_train_pipe.named_steps["clf"])
    ])

    # Final evaluation happens once, after the model and threshold are fixed.
    proba_test = prod_pipeline.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= tuned_threshold).astype(int)
    test_metrics = {
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0))
    }
    print(f"Held-out test PR-AUC: {test_metrics['pr_auc']:.3f}")
    print(f"Held-out test precision: {test_metrics['precision']:.3f}")
    print(f"Held-out test recall: {test_metrics['recall']:.3f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_no_show_model.pkl"
    threshold_path = out_dir / "threshold.json"
    joblib.dump(prod_pipeline, model_path)
    with threshold_path.open("w") as fp:
        json.dump({"threshold": tuned_threshold}, fp, indent=2)
    with (out_dir / "evaluation.json").open("w") as fp:
        json.dump({
            "selected_model": best_name,
            "training_cv_pr_auc": cv_scores,
            "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
            "seed": SEED,
            "target_validation_recall": TARGET_RECALL,
            "threshold": tuned_threshold,
            "test_metrics": test_metrics
        }, fp, indent=2)
    print(f"Saved production model to {model_path}")
    print(f"Saved threshold to {threshold_path}")
    return prod_pipeline, tuned_threshold, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a patient no-show model.")
    parser.add_argument("--data", type=Path, default=CSV_FILE,
                        help="Dataset CSV path (default: KaggleV2-May-2016.csv beside this script)")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. Place KaggleV2-May-2016.csv "
                     "beside train_model.py or pass --data /path/to/dataset.csv.")
    train_model(args.data)


if __name__ == "__main__":
    main()
