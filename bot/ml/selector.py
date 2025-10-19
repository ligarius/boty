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

from ..strategies.ensemble import Signal


@dataclass
class SelectorReport:
    accuracy: float
    f1: float
    feature_importances: Dict[str, float]

    def to_dict(self) -> Dict[str, float | Dict[str, float]]:
        """Serialize the selector report as JSON-safe primitives."""

        return {
            "accuracy": float(self.accuracy),
            "f1": float(self.f1),
            "feature_importances": {
                name: float(value) for name, value in self.feature_importances.items()
            },
        }


class SignalSelector:
    """Train and optionally persist a logistic regression model for signal quality."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=100)),
            ]
        )
        self.fitted = False

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> SelectorReport:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))
        f1 = float(f1_score(y_test, y_pred))
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        feature_importances = {feature: float(weight) for feature, weight in zip(features.columns, coefs)}
        if self.model_path:
            dump(self.pipeline, self.model_path)
        self.fitted = True
        return SelectorReport(accuracy=accuracy, f1=f1, feature_importances=feature_importances)

    def load(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must be provided to load a selector")
        self.pipeline = load(self.model_path)
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
