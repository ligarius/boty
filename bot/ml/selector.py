"""ML selector that evaluates strategies and updates ensemble weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ..strategies.ensemble import Signal


@dataclass
class SelectorReport:
    accuracy: float
    f1: float
    precision: float
    recall: float
    threshold: float
    feature_importances: Dict[str, float]

    def to_dict(self) -> Dict[str, float | Dict[str, float]]:
        """Serialize the selector report as JSON-safe primitives."""

        return {
            "accuracy": float(self.accuracy),
            "f1": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "threshold": float(self.threshold),
            "feature_importances": {
                name: float(value) for name, value in self.feature_importances.items()
            },
        }


class SignalSelector:
    """Train and optionally persist a logistic regression model for signal quality."""

    def __init__(self, model_path: str | None = None, *, min_precision: float = 0.35):
        self.model_path = model_path
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=200, class_weight="balanced"),
                ),
            ]
        )
        self.fitted = False
        self.decision_threshold = 0.5
        self.min_precision = float(np.clip(min_precision, 0.0, 1.0))

    def _fit_with_time_splits(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        n_splits: int = 5,
        assume_ordered: bool = False,
    ) -> SelectorReport:
        if not assume_ordered and not features.index.is_monotonic_increasing:
            features = features.sort_index()
            labels = labels.loc[features.index]
        else:
            labels = labels.loc[features.index]

        total_samples = len(features)
        effective_splits = max(0, min(n_splits, total_samples - 1))
        accuracies: List[float] = []
        f1_scores: List[float] = []
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        threshold_scores: List[float] = []

        if effective_splits >= 1:
            splitter = TimeSeriesSplit(n_splits=effective_splits)
            for train_index, test_index in splitter.split(features):
                if len(train_index) == 0 or len(test_index) == 0:
                    continue
                y_train = labels.iloc[train_index]
                y_test = labels.iloc[test_index]
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue
                X_train = features.iloc[train_index]
                X_test = features.iloc[test_index]
                self.pipeline.fit(X_train, y_train)
                train_scores = self.pipeline.predict_proba(X_train)[:, 1]
                threshold, _, _, _ = self._calibrate_threshold(
                    y_train, train_scores
                )
                test_scores = self.pipeline.predict_proba(X_test)[:, 1]
                y_pred = (test_scores >= threshold).astype(int)
                accuracies.append(float(np.mean(y_pred == y_test)))
                f1_scores.append(float(f1_score(y_test, y_pred, zero_division=0)))
                precision_scores.append(
                    float(precision_score(y_test, y_pred, zero_division=0))
                )
                recall_scores.append(
                    float(recall_score(y_test, y_pred, zero_division=0))
                )
                threshold_scores.append(float(threshold))

        self.pipeline.fit(features, labels)
        self.fitted = True
        all_scores = self.pipeline.predict_proba(features)[:, 1]
        threshold, precision_value, recall_value, f1_value = self._calibrate_threshold(
            labels, all_scores
        )
        self.decision_threshold = float(threshold)
        y_pred_all = (all_scores >= self.decision_threshold).astype(int)
        accuracy_raw = float(np.mean(y_pred_all == labels))
        accuracy = (
            float(np.mean(accuracies))
            if accuracies
            else accuracy_raw
        )
        f1_metric = float(np.mean(f1_scores)) if f1_scores else float(f1_value)
        precision_metric = (
            float(np.mean(precision_scores)) if precision_scores else float(precision_value)
        )
        recall_metric = (
            float(np.mean(recall_scores)) if recall_scores else float(recall_value)
        )
        threshold_metric = (
            float(np.mean(threshold_scores)) if threshold_scores else float(self.decision_threshold)
        )
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        feature_importances = {
            feature: float(weight) for feature, weight in zip(features.columns, coefs)
        }
        if self.model_path:
            dump(
                {
                    "pipeline": self.pipeline,
                    "threshold": self.decision_threshold,
                    "min_precision": self.min_precision,
                },
                self.model_path,
            )
        return SelectorReport(
            accuracy=accuracy,
            f1=f1_metric,
            precision=precision_metric,
            recall=recall_metric,
            threshold=threshold_metric,
            feature_importances=feature_importances,
        )

    def fit(
        self, features: pd.DataFrame, labels: pd.Series, *, n_splits: int = 5
    ) -> SelectorReport:
        return self._fit_with_time_splits(features, labels, n_splits=n_splits, assume_ordered=False)

    def fit_ordered(
        self, features: pd.DataFrame, labels: pd.Series, *, n_splits: int = 5
    ) -> SelectorReport:
        return self._fit_with_time_splits(features, labels, n_splits=n_splits, assume_ordered=True)

    def load(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must be provided to load a selector")
        artifact = load(self.model_path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            self.pipeline = artifact["pipeline"]
            self.decision_threshold = float(artifact.get("threshold", 0.5))
            stored_precision = artifact.get("min_precision")
            if stored_precision is not None:
                self.min_precision = float(np.clip(stored_precision, 0.0, 1.0))
        else:
            self.pipeline = artifact
            self.decision_threshold = 0.5
        self.fitted = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Selector must be fitted before predicting probabilities")
        return self.pipeline.predict_proba(features)[:, 1]

    def score_signals(self, signals: Iterable[Signal]) -> List[Signal]:
        signals_list = list(signals)
        if not signals_list:
            return []
        if not self.fitted:
            return sorted(signals_list, key=lambda sig: sig.score, reverse=True)

        features = pd.DataFrame([sig.features for sig in signals_list]).fillna(0.0)
        probabilities = self.predict_proba(features)
        threshold = float(self.decision_threshold)
        probabilities = np.where(probabilities >= threshold, probabilities, 0.0)
        enhanced: List[Signal] = []
        for sig, proba in zip(signals_list, probabilities):
            updated_features = dict(sig.features)
            updated_features["meta_probability"] = float(proba)
            enhanced.append(
                Signal(
                    symbol=sig.symbol,
                    timeframe=sig.timeframe,
                    signal=sig.signal,
                    score=float(sig.score * proba),
                    atr=sig.atr,
                    features=updated_features,
                )
            )
        return sorted(enhanced, key=lambda sig: sig.score, reverse=True)

    def _calibrate_threshold(
        self, labels: pd.Series, scores: np.ndarray
    ) -> tuple[float, float, float, float]:
        if len(scores) == 0:
            return self.decision_threshold, 0.0, 0.0, 0.0

        thresholds = np.linspace(0.1, 0.9, 17)
        best_threshold = self.decision_threshold
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0

        labels_array = labels.to_numpy()

        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            precision = precision_score(labels_array, preds, zero_division=0)
            recall = recall_score(labels_array, preds, zero_division=0)
            f1_val = f1_score(labels_array, preds, zero_division=0)
            if precision + f1_val + recall == 0:
                continue
            if precision < self.min_precision:
                continue
            if precision > best_precision + 1e-6 or (
                abs(precision - best_precision) <= 1e-6 and f1_val > best_f1 + 1e-6
            ):
                best_precision = precision
                best_recall = recall
                best_f1 = f1_val
                best_threshold = threshold

        if best_precision == 0.0 and best_f1 == 0.0:
            default_preds = (scores >= 0.5).astype(int)
            best_precision = precision_score(labels_array, default_preds, zero_division=0)
            best_recall = recall_score(labels_array, default_preds, zero_division=0)
            best_f1 = f1_score(labels_array, default_preds, zero_division=0)
            best_threshold = 0.5

        return best_threshold, best_precision, best_recall, best_f1
