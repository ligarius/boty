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


class EvolutionaryTuner:
    """Wrap Optuna study to perform evolutionary style selection."""

    def __init__(self, study_name: str, storage: str | None = None):
        self.study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")

    def optimize(self, objective: Callable[[optuna.Trial], float], n_trials: int = 20) -> CandidateResult:
        self.study.optimize(objective, n_trials=n_trials)
        best_trial = self.study.best_trial
        metrics = {"value": best_trial.value}
        metrics.update(best_trial.user_attrs)
        resolved_params = best_trial.user_attrs.get("resolved_params")
        if resolved_params is None:
            resolved_params = dict(best_trial.params)
        is_promoted = best_trial.value >= best_trial.user_attrs.get("baseline", 0.0)
        return CandidateResult(
            params=resolved_params,
            metrics=metrics,
            is_promoted=is_promoted,
        )

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, float]:
        fast = trial.suggest_int("fast", 5, 21)
        slow = trial.suggest_int("slow", 21, 55)
        threshold = trial.suggest_float("z_threshold", 1.0, 3.0)
        return {"fast": fast, "slow": slow, "z_threshold": threshold}
