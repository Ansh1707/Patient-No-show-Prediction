import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

import train_model as training


def synthetic_data():
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame({
        "Gender": rng.choice(["F", "M"], n),
        "Age": rng.integers(0, 100, n),
        "wait_days": rng.integers(0, 30, n),
        **{feature: rng.integers(0, 2, n) for feature in training.NUM_COLS
           if feature not in {"Age", "wait_days"}}
    })
    y = pd.Series((np.arange(n) % 5 == 0).astype(int), name="No-show")
    return X[training.FEATURES], y


class TrainingTests(unittest.TestCase):
    def test_splits_are_disjoint_stratified_and_reproducible(self):
        X, y = synthetic_data()
        first = training.split_data(X, y)
        second = training.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = first
        groups = [set(part.index) for part in (X_train, X_val, X_test)]
        self.assertEqual([len(group) for group in groups], [180, 60, 60])
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(set.union(*groups), set(X.index))
        for features, labels in zip(first[:3], first[3:]):
            self.assertTrue(features.index.equals(labels.index))
            self.assertAlmostEqual(labels.mean(), y.mean())
        for one, two in zip(first, second):
            self.assertTrue(one.equals(two))

    def test_highest_threshold_meeting_recall(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        proba = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.2])
        threshold = training.select_threshold(y, proba)
        self.assertEqual(threshold, 0.7)
        self.assertEqual(((proba >= threshold) & (y == 1)).sum() / 4, 0.75)
        self.assertLess(((proba >= 0.8) & (y == 1)).sum() / 4, 0.75)

    def test_tied_probabilities_and_full_recall(self):
        self.assertEqual(training.select_threshold([0, 1, 1, 0], [0.5] * 4), 0.5)
        self.assertEqual(training.select_threshold([1, 0, 1], [0.1, 0.2, 0.9], 1.0), 0.1)

    def test_loader_and_portable_default(self):
        self.assertEqual(training.CSV_FILE.parent, Path(training.__file__).resolve().parent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "appointments.csv"
            pd.DataFrame({
                "Gender": ["F", "M"], "Age": [30, -1],
                **{feature: [0, 0] for feature in training.NUM_COLS
                   if feature not in {"Age", "wait_days"}},
                "ScheduledDay": ["2016-05-01T12:00:00Z"] * 2,
                "AppointmentDay": ["2016-05-03T00:00:00Z"] * 2,
                "No-show": ["Yes", "No"]
            }).to_csv(path, index=False)
            X, y = training.load_data(path)
            self.assertEqual(len(X), 1)
            self.assertEqual(X.iloc[0]["wait_days"], 2)
            self.assertEqual(y.tolist(), [1])

    def test_only_training_is_fitted_and_only_validation_tunes_threshold(self):
        X, y = synthetic_data()
        parts = training.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = parts
        pipeline = training.build_models(y_train)["LogReg"]
        fit_indices = []
        original_fit = training.ImbPipeline.fit

        def tracked_fit(instance, features, labels, **kwargs):
            fit_indices.append(set(features.index))
            return original_fit(instance, features, labels, **kwargs)

        def serial_cv(pipe, features, labels, **kwargs):
            self.assertTrue(features.equals(X_train))
            self.assertTrue(labels.equals(y_train))
            kwargs["n_jobs"] = 1
            return cross_validate(pipe, features, labels, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(training, "load_data", return_value=(X, y)), \
                 patch.object(training, "build_models", return_value={"LogReg": pipeline}) as build, \
                 patch.object(training, "cross_validate", side_effect=serial_cv), \
                 patch.object(training.ImbPipeline, "fit", new=tracked_fit), \
                 patch.object(training, "select_threshold", wraps=training.select_threshold) as select:
                model, threshold, metrics = training.train_model("unused.csv", directory)
            self.assertTrue(build.call_args.args[0].equals(y_train))
            self.assertEqual(len(fit_indices), training.N_FOLDS + 1)
            self.assertTrue(all(rows <= set(X_train.index) for rows in fit_indices))
            self.assertEqual(fit_indices[-1], set(X_train.index))
            select.assert_called_once()
            self.assertTrue(select.call_args.args[0].equals(y_val))
            np.testing.assert_allclose(select.call_args.args[1], pipeline.predict_proba(X_val)[:, 1])
            expected_threshold = training.select_threshold(y_val, pipeline.predict_proba(X_val)[:, 1])
            self.assertEqual(threshold, expected_threshold)
            # The exported pipeline must retain the model whose threshold was tuned.
            saved = joblib.load(Path(directory) / "best_no_show_model.pkl")
            np.testing.assert_allclose(saved.predict_proba(X_test), pipeline.predict_proba(X_test))
            self.assertNotIn("smote", saved.named_steps)
            expected_proba = saved.predict_proba(X_test)[:, 1]
            self.assertAlmostEqual(metrics["pr_auc"], training.average_precision_score(y_test, expected_proba))
            self.assertAlmostEqual(metrics["recall"], training.recall_score(y_test, expected_proba >= threshold))
            self.assertAlmostEqual(metrics["precision"], training.precision_score(
                y_test, expected_proba >= threshold, zero_division=0))
            with (Path(directory) / "threshold.json").open() as file:
                self.assertEqual(json.load(file)["threshold"], threshold)
            with (Path(directory) / "evaluation.json").open() as file:
                report = json.load(file)
            self.assertEqual(report["test_metrics"], metrics)
            self.assertEqual(report["split_sizes"], {"train": 180, "validation": 60, "test": 60})


if __name__ == "__main__":
    unittest.main()
