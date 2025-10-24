"""Backtesting utilities powered by vectorbt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import vectorbt as vbt
except ImportError:  # pragma: no cover - fallback for environments without vectorbt
    vbt = None  # type: ignore

from ..core.config import get_settings
from ..core.risk import RiskManager
from ..ml.selector import SelectorReport, SignalSelector
from ..strategies import mean_reversion, momentum, volatility_breakout, market_making
from ..strategies.ensemble import Signal
from sklearn.metrics import f1_score


@dataclass
class BacktestMetrics:
    roi: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    training_accuracy: float | None = None
    training_f1: float | None = None
    training_precision: float | None = None
    training_recall: float | None = None
    selector_threshold: float | None = None

    def to_dict(self) -> Dict[str, float]:
        """Serialize metrics as a plain dictionary."""

        return {
            "roi": float(self.roi),
            "sharpe": float(self.sharpe),
            "max_drawdown": float(self.max_drawdown),
            "profit_factor": float(self.profit_factor),
            "win_rate": float(self.win_rate),
        }


@dataclass
class _SelectorOptimizationResult:
    probabilities: pd.Series
    reports: List[SelectorReport]
    resolved_window: int
    resolved_threshold: float


class BacktestEngine:
    """Run walk-forward backtests and compute metrics."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.risk = RiskManager(self.settings)
        self.selector_threshold = self.settings.selector_threshold
        self.selector_horizon = self.settings.selector_horizon
        self.selector_window = self.settings.selector_window
        self.selector_auto_optimize = getattr(self.settings, "selector_auto_optimize", False)
        self.last_training_report: SelectorReport | None = None
        self.last_signal_probabilities: pd.DataFrame | None = None
        self.last_weighted_scores: pd.Series | None = None
        self.last_weighted_scores_raw: pd.Series | None = None
        self.last_signals: List[Signal] = []
        self.last_resolved_selector_window: int | None = None
        self.last_resolved_selector_threshold: float | None = None

    @staticmethod
    def _sanitize(value: float) -> float:
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
        return 0.0

    def _build_training_samples(
        self,
        data: pd.DataFrame,
        momentum_df: pd.DataFrame,
        mean_df: pd.DataFrame,
        breakout_df: pd.DataFrame,
        mm_df: pd.DataFrame,
        effective_timeframe: str,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[Signal]]:
        future_returns = data["close"].pct_change(periods=self.selector_horizon).shift(-self.selector_horizon)
        feature_sets: List[Dict[str, float]] = []
        labels: List[int] = []
        metadata_rows: List[Dict[str, object]] = []

        symbol = self.settings.universe[0] if getattr(self.settings, "universe", None) else "UNKNOWN"

        resolved_timeframe = effective_timeframe or "UNKNOWN"

        strategies: List[Tuple[str, pd.DataFrame, float]] = [
            ("momentum", momentum_df, 0.0),
            ("mean_reversion", mean_df, 1.0),
            ("volatility_breakout", breakout_df, 2.0),
            ("market_making", mm_df, 3.0),
        ]

        for name, df, strategy_id in strategies:
            df = df.copy()
            feature_cols = [col for col in df.columns if col not in {"signal", "score"}]
            if feature_cols:
                feature_ready = df[feature_cols].notna().all(axis=1)
            else:
                feature_ready = pd.Series(True, index=df.index)
            signal_series = df.get("signal", pd.Series(0, index=df.index)).fillna(0).astype(int)
            mask = (signal_series != 0) & feature_ready & future_returns.notna()
            if not mask.any():
                continue
            selected = df.loc[mask, feature_cols + ["signal", "score"]].copy()
            selected["strategy_id"] = float(strategy_id)
            selected["future_return"] = future_returns.loc[selected.index]
            selected["label"] = (selected["future_return"] * selected["signal"] > 0).astype(int)
            selected_features = selected[feature_cols + ["strategy_id"]].astype(float)
            feature_sets.extend(selected_features.to_dict(orient="records"))
            labels.extend(selected["label"].astype(int).tolist())

            timestamps = list(selected.index)
            scores = selected["score"].astype(float).tolist()
            signals = selected["signal"].astype(int).tolist()
            if "atr" in selected.columns:
                atrs = selected["atr"].astype(float).tolist()
            else:
                atrs = [float(np.nan)] * len(selected)
            for ts, sig_val, score_val, atr_val in zip(timestamps, signals, scores, atrs):
                metadata_rows.append(
                    {
                        "timestamp": ts,
                        "signal": sig_val,
                        "score": score_val,
                        "atr": atr_val,
                        "source": name,
                        "strategy": name,
                    }
                )

        if not feature_sets:
            empty_metadata = pd.DataFrame(columns=["timestamp", "signal", "score", "atr", "source"])
            return pd.DataFrame(), pd.Series(dtype=int), empty_metadata, []

        features_df = pd.DataFrame(feature_sets).fillna(0.0)
        labels_series = pd.Series(labels, index=features_df.index, dtype=int)
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df.index = features_df.index
        metadata_df["score"] = metadata_df["score"].astype(float)
        metadata_df["signal"] = metadata_df["signal"].astype(int)
        metadata_df["atr"] = pd.to_numeric(metadata_df["atr"], errors="coerce")

        signals_objects: List[Signal] = []
        for idx, feature_row in features_df.iterrows():
            meta = metadata_df.loc[idx]
            feature_dict = {col: float(feature_row[col]) for col in features_df.columns}
            atr_value = float(meta["atr"]) if pd.notna(meta["atr"]) else float(np.nan)
            signals_objects.append(
                Signal(
                    symbol=symbol,
                    timeframe=resolved_timeframe,
                    signal=int(meta["signal"]),
                    score=float(meta["score"]),
                    atr=atr_value,
                    features=feature_dict,
                    strategy=str(meta.get("strategy")) if "strategy" in meta else None,
                )
            )

        return features_df, labels_series, metadata_df, signals_objects

    def _walk_forward_probabilities(
        self,
        sorted_features: pd.DataFrame,
        sorted_labels: pd.Series,
        window_size: int,
    ) -> Tuple[pd.Series, List[SelectorReport]]:
        probabilities = pd.Series(np.nan, index=sorted_features.index, dtype=float)
        reports: List[SelectorReport] = []

        for start in range(0, len(sorted_features), window_size):
            end = min(start + window_size, len(sorted_features))
            window_indices = sorted_features.index[start:end]
            train_indices = sorted_features.index[:start]
            if len(train_indices) < 2 or sorted_labels.loc[train_indices].nunique() < 2:
                probabilities.loc[window_indices] = 1.0
                continue

            selector = SignalSelector(model_path=None)
            report = selector.fit_ordered(
                sorted_features.loc[train_indices],
                sorted_labels.loc[train_indices],
            )
            reports.append(report)
            window_probas = selector.predict_proba(sorted_features.loc[window_indices])
            threshold = float(getattr(selector, "decision_threshold", 0.5))
            adjusted = np.where(window_probas >= threshold, window_probas, 0.0)
            probabilities.loc[window_indices] = adjusted

        probabilities = probabilities.fillna(1.0)
        return probabilities, reports

    def _auto_optimize_selector_params(
        self,
        sorted_features: pd.DataFrame,
        sorted_labels: pd.Series,
        base_window: int,
    ) -> Optional[_SelectorOptimizationResult]:
        if not self.selector_auto_optimize:
            return None
        if len(sorted_features) < 5 or sorted_labels.nunique() < 2:
            return None

        max_window = max(1, len(sorted_features))
        base_window = min(max_window, max(1, base_window))

        candidate_windows = {
            base_window,
            max(1, min(max_window, base_window // 2)),
            max(1, min(max_window, int(base_window * 1.5))),
            max(1, min(max_window, int(len(sorted_features) * 0.25))),
            max(1, min(max_window, int(len(sorted_features) * 0.5))),
        }
        candidate_windows = {w for w in candidate_windows if w >= 1}
        if not candidate_windows:
            candidate_windows = {base_window}

        base_threshold = float(self.selector_threshold)
        threshold_grid = [round(x, 4) for x in np.linspace(0.02, 0.5, 10)]
        threshold_candidates = {base_threshold, *threshold_grid}
        threshold_candidates = {th for th in threshold_candidates if 0.0 <= th <= 1.0}

        probability_cache: Dict[int, Tuple[pd.Series, List[SelectorReport]]] = {}

        def compute(window: int) -> Tuple[pd.Series, List[SelectorReport]]:
            window = max(1, min(max_window, window))
            if window not in probability_cache:
                probability_cache[window] = self._walk_forward_probabilities(
                    sorted_features, sorted_labels, window
                )
            return probability_cache[window]

        baseline_probs, baseline_reports = compute(base_window)
        baseline_pred = (baseline_probs >= base_threshold).astype(int)
        baseline_f1 = float(
            f1_score(sorted_labels.values, baseline_pred.values, zero_division=0)
        )
        baseline_acc = float((baseline_pred.values == sorted_labels.values).mean())

        best_result = _SelectorOptimizationResult(
            probabilities=baseline_probs,
            reports=baseline_reports,
            resolved_window=base_window,
            resolved_threshold=base_threshold,
        )
        best_f1 = baseline_f1
        best_acc = baseline_acc

        for window in sorted(candidate_windows):
            probs, reports = compute(window)
            if probs.empty:
                continue
            for threshold in threshold_candidates:
                pred = (probs >= threshold).astype(int)
                f1_val = float(
                    f1_score(sorted_labels.values, pred.values, zero_division=0)
                )
                acc_val = float((pred.values == sorted_labels.values).mean())
                if f1_val > best_f1 + 1e-6 or (
                    abs(f1_val - best_f1) <= 1e-6 and acc_val > best_acc + 1e-6
                ):
                    best_f1 = f1_val
                    best_acc = acc_val
                    best_result = _SelectorOptimizationResult(
                        probabilities=probs,
                        reports=reports,
                        resolved_window=window,
                        resolved_threshold=threshold,
                    )

        if (
            best_result.resolved_window != base_window
            or abs(best_result.resolved_threshold - base_threshold) > 1e-6
        ):
            self.selector_window = best_result.resolved_window
            self.selector_threshold = best_result.resolved_threshold

        return best_result

    def _train_selector(
        self, features: pd.DataFrame, labels: pd.Series, metadata: pd.DataFrame
    ) -> Tuple[SelectorReport | None, pd.Series, int, float]:
        final_window = max(int(self.selector_window), 1)
        final_threshold = float(self.selector_threshold)

        if features.empty or len(features) < 2 or labels.nunique() < 2:
            if features.empty:
                return None, pd.Series(dtype=float), final_window, final_threshold
            baseline = pd.Series(np.ones(len(features)), index=features.index, dtype=float)
            return None, baseline, final_window, final_threshold

        if "timestamp" not in metadata.columns:
            raise ValueError("metadata must contain a 'timestamp' column for walk-forward training")

        sortable_metadata = metadata.loc[features.index].copy()
        sortable_metadata["timestamp"] = pd.to_datetime(sortable_metadata["timestamp"])
        sortable_metadata["__order"] = np.arange(len(sortable_metadata))
        sort_columns = ["timestamp"]
        if "source" in sortable_metadata.columns:
            sort_columns.append("source")
        sort_columns.append("__order")
        ordered_indices = sortable_metadata.sort_values(sort_columns).index

        sorted_features = features.loc[ordered_indices]
        sorted_labels = labels.loc[ordered_indices]

        window_size = final_window
        optimization = self._auto_optimize_selector_params(
            sorted_features, sorted_labels, window_size
        )

        if optimization is not None:
            window_size = optimization.resolved_window
            probabilities = optimization.probabilities.copy()
            reports = optimization.reports
            final_window = int(optimization.resolved_window)
            final_threshold = float(optimization.resolved_threshold)
        else:
            probabilities, reports = self._walk_forward_probabilities(
                sorted_features, sorted_labels, window_size
            )
            final_window = int(max(1, window_size))
            final_threshold = float(self.selector_threshold)

        probabilities = probabilities.reindex(features.index).fillna(1.0)

        report: SelectorReport | None = reports[-1] if reports else None
        return report, probabilities, final_window, final_threshold

    def _resolve_timeframe(self, data: pd.DataFrame, timeframe: str | None) -> str:
        if timeframe:
            return timeframe
        attr_timeframe = None
        if hasattr(data, "attrs"):
            attr_timeframe = data.attrs.get("timeframe")
        if isinstance(attr_timeframe, str) and attr_timeframe:
            return attr_timeframe

        index = getattr(data, "index", None)
        freq: str | None = None
        if index is not None:
            freq = getattr(index, "freqstr", None)
            if not freq:
                freq = getattr(index, "inferred_freq", None)
        if isinstance(freq, str) and freq:
            return freq

        timeframes = getattr(self.settings, "timeframes", None)
        if isinstance(timeframes, list) and timeframes:
            first = timeframes[0]
            if isinstance(first, str) and first:
                return first
        return "UNKNOWN"

    def run(
        self,
        data: pd.DataFrame,
        timeframe: str | None = None,
        *,
        momentum_params: Dict[str, float] | None = None,
        mean_reversion_params: Dict[str, float] | None = None,
        volatility_params: Dict[str, float] | None = None,
        market_making_params: Dict[str, float] | None = None,
    ) -> BacktestMetrics:
        effective_timeframe = self._resolve_timeframe(data, timeframe)
        momentum_params = momentum_params or {}
        mean_reversion_params = mean_reversion_params or {}
        volatility_params = volatility_params or {}
        market_making_params = market_making_params or {}

        momentum_df = momentum.momentum_signals(data, **momentum_params).reindex(data.index)
        mean_df = mean_reversion.mean_reversion_signals(data, **mean_reversion_params).reindex(data.index)
        breakout_df = volatility_breakout.volatility_breakout_signals(data, **volatility_params).reindex(
            data.index
        )
        mm_df = market_making.market_making_signals(data, **market_making_params).reindex(data.index)

        features, labels, signal_metadata, signals = self._build_training_samples(
            data,
            momentum_df,
            mean_df,
            breakout_df,
            mm_df,
            effective_timeframe,
        )
        (
            report,
            probabilities,
            resolved_selector_window,
            resolved_selector_threshold,
        ) = self._train_selector(features, labels, signal_metadata)
        self.last_training_report = report
        self.last_signals = signals
        self.last_resolved_selector_window = int(resolved_selector_window)
        self.last_resolved_selector_threshold = float(resolved_selector_threshold)

        probability_table = signal_metadata.sort_values("timestamp").copy()
        if probability_table.empty:
            probability_table["probability"] = pd.Series(dtype=float)
            probability_table["weighted_score"] = pd.Series(dtype=float)
            weighted_scores = (
                momentum_df["score"].fillna(0.0) + mean_df["score"].fillna(0.0)
            ).reindex(data.index, fill_value=0.0)
        else:
            proba_series = probabilities.reindex(probability_table.index)
            if proba_series.empty:
                proba_series = pd.Series(
                    np.ones(len(probability_table)), index=probability_table.index, dtype=float
                )
            probability_table["probability"] = proba_series.fillna(1.0).clip(0.0, 1.0)
            probability_table["weighted_score"] = (
                probability_table["score"] * probability_table["probability"]
            )
            weighted_scores = (
                probability_table.groupby("timestamp")["weighted_score"].sum().reindex(
                    data.index, fill_value=0.0
                )
            )

        self.last_signal_probabilities = probability_table
        raw_weighted_scores = weighted_scores.astype(float)
        threshold = float(resolved_selector_threshold)
        if threshold > 0:
            clip_limit = float(np.nextafter(threshold, 0.0))
            raw_weighted_scores = raw_weighted_scores.clip(-clip_limit, clip_limit)
        rescaled_reference = raw_weighted_scores.copy()
        abs_max = raw_weighted_scores.abs().max()
        if abs_max and abs_max > 0:
            normalized_weighted_scores = raw_weighted_scores / abs_max
        else:
            normalized_weighted_scores = raw_weighted_scores.copy()

        final_signal = pd.Series(0, index=data.index, dtype=int)
        final_signal.loc[normalized_weighted_scores > threshold] = 1
        final_signal.loc[normalized_weighted_scores < -threshold] = -1

        momentum_ready = momentum_df.drop(columns=["signal", "score"], errors="ignore").notna().all(axis=1)
        mean_ready = mean_df.drop(columns=["signal", "score"], errors="ignore").notna().all(axis=1)
        valid_mask = (momentum_ready & mean_ready).reindex(data.index, fill_value=False)
        if not valid_mask.any():
            valid_mask = pd.Series(True, index=data.index)

        final_signal = final_signal.where(valid_mask, 0)
        normalized_weighted_scores = normalized_weighted_scores.where(valid_mask, 0.0)

        self.last_weighted_scores = normalized_weighted_scores
        self.last_weighted_scores_raw = rescaled_reference

        price = data.loc[valid_mask, "close"]
        final_valid = final_signal.loc[valid_mask]
        prev_signal = final_valid.shift(fill_value=0)
        long_entries = (final_valid == 1) & (prev_signal <= 0)
        long_exits = (final_valid <= 0) & (prev_signal == 1)
        short_entries = (final_valid == -1) & (prev_signal >= 0)
        short_exits = (final_valid >= 0) & (prev_signal == -1)

        training_accuracy = report.accuracy if report else None
        training_f1 = report.f1 if report else None
        training_precision = report.precision if report else None
        training_recall = report.recall if report else None
        calibrated_threshold = (
            float(report.threshold) if report else float(resolved_selector_threshold)
        )

        if vbt is not None:
            pf = vbt.Portfolio.from_signals(
                price,
                entries=long_entries.astype(bool),
                exits=long_exits.astype(bool),
                short_entries=short_entries.astype(bool),
                short_exits=short_exits.astype(bool),
                fees=0.0004,
                sl_stop=0.01,
                tp_stop=0.02,
            )
            stats = pf.stats()
            return BacktestMetrics(
                roi=self._sanitize(float(stats.loc["Total Return [%]"]) / 100),
                sharpe=self._sanitize(float(stats.loc["Sharpe Ratio"])),
                max_drawdown=self._sanitize(float(stats.loc["Max Drawdown [%]"]) / 100),
                profit_factor=self._sanitize(float(stats.loc["Profit Factor"])),
                win_rate=self._sanitize(float(stats.loc["Win Rate [%]"]) / 100),
                training_accuracy=training_accuracy,
                training_f1=training_f1,
                training_precision=training_precision,
                training_recall=training_recall,
                selector_threshold=calibrated_threshold,
            )

        # numpy fallback when vectorbt is unavailable
        trimmed_price = data.loc[valid_mask, "close"]
        returns = trimmed_price.pct_change().fillna(0.0)
        position = final_valid.replace(0, np.nan).ffill().fillna(0)
        strategy_returns = returns * position.shift(fill_value=0)

        if strategy_returns.empty:
            roi = 0.0
            sharpe = 0.0
            max_drawdown = 0.0
            profit_factor = 1.0
            win_rate = 0.0
        else:
            equity_curve = (1 + strategy_returns).cumprod()
            roi = float(equity_curve.iloc[-1] - 1)
            sharpe = float(
                strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 24 * 60)
            )
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve / rolling_max - 1).min()
            max_drawdown = float(-drawdown)
            gains = strategy_returns[strategy_returns > 0].sum()
            losses = -strategy_returns[strategy_returns < 0].sum()
            profit_factor = float(np.maximum(gains, 1e-6) / np.maximum(losses, 1e-6))
            win_rate = float((strategy_returns > 0).mean())

        return BacktestMetrics(
            roi=self._sanitize(roi),
            sharpe=self._sanitize(sharpe),
            max_drawdown=self._sanitize(max_drawdown),
            profit_factor=self._sanitize(profit_factor),
            win_rate=self._sanitize(win_rate),
            training_accuracy=training_accuracy,
            training_f1=training_f1,
            training_precision=training_precision,
            training_recall=training_recall,
            selector_threshold=calibrated_threshold,
        )

    def meets_go_live(self, metrics: BacktestMetrics) -> bool:
        return (
            metrics.sharpe >= 1.2
            and metrics.profit_factor >= 1.3
            and metrics.max_drawdown <= 0.08
            and metrics.win_rate >= 0.45
        )
