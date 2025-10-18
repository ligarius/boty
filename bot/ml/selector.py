"""ML selector that evaluates strategies and updates ensemble weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ..strategies.ensemble import Signal, EnsembleSelector


@dataclass
class SelectorReport:
    accuracy: float
    f1: float
    feature_importances: Dict[str, float]


class SignalSelector:
    """Train and persist a logistic regression model for signal quality."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=100)),
            ]
        )
        self.fitted = False

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> SelectorReport:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))
        f1 = float(f1_score(y_test, y_pred))
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        feature_importances = {feature: float(weight) for feature, weight in zip(features.columns, coefs)}
        dump(self.pipeline, self.model_path)
        self.fitted = True
        return SelectorReport(accuracy=accuracy, f1=f1, feature_importances=feature_importances)

    def load(self) -> None:
        self.pipeline = load(self.model_path)
        self.fitted = True

    def score_signals(self, signals: Iterable[Signal]) -> List[Signal]:
        selector = EnsembleSelector()
        if self.fitted:
            selector.model = self.pipeline
            selector.is_trained = True
        return selector.score_signals(signals)
