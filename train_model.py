import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Config
CSV_FILE        = "/Users/ansh/Desktop/UPES/Internship/no-show-web/KaggleV2-May-2016.csv"  
SEED            = 42
TEST_SIZE       = 0.20
N_FOLDS         = 5
TARGET_RECALL   = 0.75
SELECTION_METRIC = "pr_auc"
OUT_DIR         = Path("model")
OUT_DIR.mkdir(exist_ok=True)

# 1. LOAD & CLEAN (Neighbourhood removed)
df = pd.read_csv(CSV_FILE, parse_dates=["ScheduledDay", "AppointmentDay"])
df["wait_days"] = (df["AppointmentDay"].dt.date - df["ScheduledDay"].dt.date).dt.days
df = df[df.Age >= 0]

X = df[["Gender", "Age", "Scholarship", "Hipertension",
        "Diabetes", "Alcoholism", "Handcap", "SMS_received", "wait_days"]]
y = (df["No-show"] == "Yes").astype(int)  # 1 = no-show, 0 = show

# 2. PREPROCESSOR (only Gender categorical)
cat_cols = ["Gender"]
num_cols = ["Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
            "Handcap", "SMS_received", "wait_days"]

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preproc = ColumnTransformer([
    ("ohe", ohe, cat_cols),
    ("scale", StandardScaler(), num_cols)
])

# 3. PIPELINE WITH SMOTE (for training CV only)
def make_training_pipe(clf):
    return ImbPipeline([
        ("prep", preproc),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", clf)
    ])

neg, pos = np.bincount(y)
scale_pos = neg / pos

models = {
    "LogReg": make_training_pipe(
        LogisticRegression(class_weight="balanced", solver="liblinear", random_state=SEED)),
    "RF": make_training_pipe(
        RandomForestClassifier(class_weight="balanced_subsample", n_estimators=300, n_jobs=-1,
                               random_state=SEED)),
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

scoring = {"pr_auc": "average_precision"}

# 4. TRAIN / CV
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_scores = {}
for name, pipe in models.items():
    res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
    cv_scores[name] = res["test_pr_auc"].mean()

best_name = max(cv_scores, key=cv_scores.get)
best_train_pipe = models[best_name]
print(f"Best model by PR-AUC: {best_name} ({cv_scores[best_name]:.3f})")

# Fit best model on full training split
best_train_pipe.fit(X_tr, y_tr)

# 5. THRESHOLD TUNING WITH CORRECT INDEXING
proba_te = best_train_pipe.predict_proba(X_te)[:, 1]
prec, rec, thr = precision_recall_curve(y_te, proba_te)

print(f"\nRecall and thresholds:")
for i in range(len(thr)):
    print(f"{i:3d}: Recall={rec[i]:.4f}, Threshold={thr[i]:.4f}")

rec_targets = np.where(rec[:-1] >= TARGET_RECALL)[0]  # note thr length = rec length - 1
if len(rec_targets) > 0:
    idx = rec_targets[-1]
    tuned_threshold = thr[idx]
else:
    tuned_threshold = 0.5

print(f"\nSelected threshold for recall ≥ {TARGET_RECALL}: {tuned_threshold:.3f}")
print(f"PR-AUC: {average_precision_score(y_te, proba_te):.3f}")

# 6. PRODUCTION PIPELINE WITHOUT SMOTE
prod_pipeline = SkPipeline([
    ("prep", preproc),
    ("clf", best_train_pipe.named_steps["clf"])
])
prod_pipeline.fit(X, y)

model_path = OUT_DIR / "best_no_show_model.pkl"
threshold_path = OUT_DIR / "threshold.json"
joblib.dump(prod_pipeline, model_path)
with open(threshold_path, "w") as fp:
    json.dump({"threshold": float(tuned_threshold)}, fp, indent=2)

print(f"✔ Saved production model to {model_path}")
print(f"✔ Saved threshold to {threshold_path}")

# 7. Feature No-Show Rate for Reference
print("\nNo-show rate by feature (value=1 vs 0):")
for feat in ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received"]:
    print(f"{feat} = 1 -> {df[df[feat] == 1]['No-show'].eq('Yes').mean():.3f}")
    print(f"{feat} = 0 -> {df[df[feat] == 0]['No-show'].eq('Yes').mean():.3f}")
