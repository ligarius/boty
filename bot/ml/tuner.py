"""Evolutionary hyperparameter tuner leveraging Optuna."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import optuna


@dataclass
class CandidateResult:
    params: Dict[str, float]
    metrics: Dict[str, float]
    is_promoted: bool
    passes_thresholds: bool


class EvolutionaryTuner:
    """Wrap Optuna study to perform evolutionary style selection."""

    def __init__(
        self,
        study_name: str,
        storage: str | None = None,
        *,
        min_roi: float = 0.0,
        min_profit_factor: float = 1.0,
        max_drawdown: float = 0.35,
    ):
        self.study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")
        self.min_roi = min_roi
        self.min_profit_factor = min_profit_factor
        self.max_drawdown = max_drawdown

    def optimize(self, objective: Callable[[optuna.Trial], float], n_trials: int = 20) -> CandidateResult:
        self.study.optimize(objective, n_trials=n_trials)
        best_trial = self.study.best_trial
        metrics = {"value": best_trial.value}
        metrics.update(best_trial.user_attrs)
        resolved_params = dict(best_trial.user_attrs.get("resolved_params", {}))
        if not resolved_params:
            resolved_params = dict(best_trial.params)

        resolved_window = best_trial.user_attrs.get("resolved_selector_window")
        if resolved_window is not None:
            resolved_params["selector_window"] = int(resolved_window)

        resolved_threshold = best_trial.user_attrs.get("resolved_selector_threshold")
        if resolved_threshold is not None:
            resolved_params["selector_threshold"] = float(resolved_threshold)
        metrics_payload = best_trial.user_attrs.get("metrics", {})
        passes_thresholds = self._passes_thresholds(metrics_payload)
        is_promoted = best_trial.value >= best_trial.user_attrs.get("baseline", 0.0) and passes_thresholds
        return CandidateResult(
            params=resolved_params,
            metrics=metrics,
            is_promoted=is_promoted,
            passes_thresholds=passes_thresholds,
        )

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, float]:
        fast = trial.suggest_int("fast", 5, 21)
        slow = trial.suggest_int("slow", 21, 55)
        threshold = trial.suggest_float("z_threshold", 1.0, 3.0)
        return {"fast": fast, "slow": slow, "z_threshold": threshold}

    def _passes_thresholds(self, metrics: Dict[str, float]) -> bool:
        if not metrics:
            return False
        roi = float(metrics.get("roi", -1.0))
        profit_factor = float(metrics.get("profit_factor", 0.0))
        drawdown = float(metrics.get("max_drawdown", 1.0))
        return (
            roi >= self.min_roi
            and profit_factor >= self.min_profit_factor
            and drawdown <= self.max_drawdown
        )
