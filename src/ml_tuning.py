"""Подбор параметров HGB только внутри train и отдельная оценка выбранного варианта."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid, ParameterSampler
from threadpoolctl import threadpool_limits

from .data_validation import validate_power_series
from .ml_baselines import calculate_forecast_metrics
from .ml_boosting import BOOSTING_METHOD, BoostingExperiment, run_boosting_experiment
from .ml_features import HISTORY_DAYS, build_day_ahead_features


TUNED_METHOD = "hist_gradient_boosting_tuned"
TUNED_LABEL = "Бустинг 3.1: подбор"
ORIGINAL_LABEL = "Бустинг 3: исходный"
TUNABLE_PARAMETERS = {"max_leaf_nodes", "min_samples_leaf", "learning_rate", "max_iter", "l2_regularization"}


@dataclass
class BoostingSearch:
    summary: pd.DataFrame
    fold_scores: pd.DataFrame
    folds: pd.DataFrame
    oof_predictions: pd.DataFrame
    best_candidate_id: str
    best_parameters: dict
    base_parameters: dict
    training_digest: str
    feature_names: tuple[str, ...]


@dataclass
class TunedBoostingExperiment:
    search: BoostingSearch
    selected: BoostingExperiment
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    daily_mae: pd.DataFrame


# Связывает результат поиска с конкретными временными метками и значениями обучения.
def _training_digest(train: pd.Series) -> str:
    values = pd.util.hash_pandas_object(train, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


# Включает исходную модель и воспроизводимо выбирает остальные уникальные конфигурации.
def make_boosting_candidates(base_parameters: dict, search_space: dict,
                             n_candidates: int = 25, random_state: int = 42) -> dict[str, dict]:
    if n_candidates < 1:
        raise ValueError("Нужна хотя бы одна конфигурация.")
    if base_parameters.get("early_stopping", "auto") is not False:
        raise ValueError("Ранняя остановка должна быть отключена во всех конфигурациях.")
    if not search_space or not set(search_space).issubset(TUNABLE_PARAMETERS):
        raise ValueError("Поиск разрешён только по пяти заявленным параметрам HGB.")
    total = len(ParameterGrid(search_space))
    candidates = {"original": dict(base_parameters)}
    # Запас в одну комбинацию позволяет исключить совпадение с исходной моделью.
    samples = ParameterSampler(search_space, n_iter=min(total, n_candidates), random_state=random_state)
    for sample in samples:
        if len(candidates) >= n_candidates:
            break
        parameters = {**base_parameters, **sample}
        if parameters not in candidates.values():
            candidates[f"C{len(candidates):02d}"] = parameters
    return candidates


# Проверяет полные, упорядоченные и непересекающиеся блоки, целиком лежащие внутри train.
def make_temporal_folds(train: pd.Series, windows, interval_minutes: int = 15) -> pd.DataFrame:
    validate_power_series(train, interval_minutes, complete_days=True)
    step = pd.Timedelta(minutes=interval_minutes)
    first_target = train.index[0] + pd.Timedelta(days=HISTORY_DAYS)
    previous_end = None
    rows = []
    for number, (start, end) in enumerate(windows, 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if (pd.isna(start) or pd.isna(end) or start.tzinfo is not None or end.tzinfo is not None
                or start != start.normalize() or end != end.normalize() or start >= end):
            raise ValueError("Границы блока должны задавать полные локальные сутки.")
        if start <= first_target or end > train.index[-1] + step:
            raise ValueError("Блок и его обучающие цели должны целиком находиться внутри train.")
        if previous_end is not None and start < previous_end:
            raise ValueError("Проверяемые блоки должны идти по времени без пересечений.")
        fitting = train.loc[(train.index >= first_target) & (train.index < start)]
        checking = train.loc[(train.index >= start) & (train.index < end)]
        expected = pd.date_range(start, end, freq=f"{interval_minutes}min", inclusive="left")
        if fitting.empty or not checking.index.equals(expected):
            raise ValueError("Для каждого блока нужны обучающие цели и все проверяемые интервалы.")
        rows.append({"fold": number, "train_start": fitting.index[0], "train_end": fitting.index[-1],
                     "validation_start": start, "validation_end": checking.index[-1],
                     "end_exclusive": end, "train_intervals": len(fitting), "intervals": len(checking)})
        previous_end = end
    if not rows:
        raise ValueError("Нужен хотя бы один временной блок.")
    return pd.DataFrame(rows).set_index("fold")


# Объединяет ошибки по всем интервалам; RMSE и WAPE не усредняются по блокам.
def aggregate_cv_scores(fold_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, scores in fold_scores.groupby("candidate", sort=False):
        count = scores["intervals"].sum()
        absolute_error = (scores["mae_kw"] * scores["intervals"]).sum()
        squared_error = (scores["rmse_kw"] ** 2 * scores["intervals"]).sum()
        denominator = scores["absolute_actual_sum"].sum()
        rows.append({"candidate": candidate, "cv_mae_kw": absolute_error / count,
                     "cv_rmse_kw": np.sqrt(squared_error / count),
                     "cv_wape_pct": 100 * absolute_error / denominator if denominator else np.nan,
                     "worst_fold_mae_kw": scores["mae_kw"].max(), "intervals": count})
    return pd.DataFrame(rows).set_index("candidate").sort_values("cv_mae_kw", kind="stable")


# Записывает ход долгого расчёта в игнорируемую папку вместо длинного вывода в ноутбуке.
def _save_search_progress(output_dir, state, started):
    if output_dir is None:
        return
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "progress.json.part"
    temporary.write_text(json.dumps({**state, "elapsed_seconds": round(time.monotonic() - started, 1)}),
                         encoding="utf-8")
    temporary.replace(path / "progress.json")


# Получает только train: данные внешней валидации и теста не участвуют в выборе параметров.
def search_boosting_parameters(train: pd.Series, base_parameters: dict, search_space: dict,
                               windows, n_candidates: int = 25, random_state: int = 42,
                               interval_minutes: int = 15, output_dir=None) -> BoostingSearch:
    folds = make_temporal_folds(train, windows, interval_minutes)
    candidates = make_boosting_candidates(base_parameters, search_space, n_candidates, random_state)
    features, _ = build_day_ahead_features(train, interval_minutes)
    eligible = features.index[HISTORY_DAYS * (1440 // interval_minutes):]
    if not np.isfinite(features.loc[eligible].to_numpy()).all():
        raise ValueError("Обучающие признаки после прогрева должны быть конечными.")
    blocks = []
    for fold, row in folds.iterrows():
        fit_index = eligible[eligible < row.validation_start]
        check_index = train.index[(train.index >= row.validation_start) & (train.index < row.end_exclusive)]
        blocks.append((fold, fit_index, check_index))
    evaluation_index = blocks[0][2].append([block[2] for block in blocks[1:]])
    oof = train.reindex(evaluation_index).to_frame("actual_kw")
    records, completed = [], 0
    started = time.monotonic()
    planned = len(candidates) * len(blocks)
    with threadpool_limits(limits=1):
        for candidate, parameters in candidates.items():
            for fold, fit_index, check_index in blocks:
                _save_search_progress(output_dir, {"phase": "fitting", "candidate": candidate, "fold": int(fold),
                                      "completed_fits": completed, "planned_fits": planned}, started)
                model = HistGradientBoostingRegressor(**parameters)
                model.fit(features.loc[fit_index], train.loc[fit_index])
                prediction = pd.Series(model.predict(features.loc[check_index]), index=check_index)
                metrics = calculate_forecast_metrics(train.loc[check_index], prediction)
                oof.loc[check_index, candidate] = prediction
                records.append({"candidate": candidate, "fold": fold, "intervals": len(check_index),
                                "absolute_actual_sum": train.loc[check_index].abs().sum(), **metrics.to_dict()})
                completed += 1
            if output_dir is not None:
                pd.DataFrame(records).to_csv(Path(output_dir) / "cv_fold_scores.csv", index=False)
    fold_scores = pd.DataFrame(records)
    if oof.isna().any().any():
        raise AssertionError("Каждая конфигурация должна покрывать одинаковые проверяемые интервалы.")
    summary = aggregate_cv_scores(fold_scores).join(pd.DataFrame.from_dict(candidates, orient="index"))
    best = summary.index[0]
    result = BoostingSearch(summary, fold_scores, folds, oof, best, dict(candidates[best]),
                            dict(base_parameters), _training_digest(train), tuple(features.columns))
    if output_dir is not None:
        save_boosting_search(result, output_dir)
    _save_search_progress(output_dir, {"phase": "complete", "completed_fits": completed,
                          "planned_fits": planned, "best_candidate": best,
                          "best_cv_mae_kw": float(summary.iloc[0].cv_mae_kw)}, started)
    return result


# Обучает выбранную по CV конфигурацию на полном train и один раз оценивает её на 2014 году.
def evaluate_tuned_boosting(train: pd.Series, validation: pd.Series, search: BoostingSearch,
                            reference: BoostingExperiment, interval_minutes: int = 15) -> TunedBoostingExperiment:
    if _training_digest(train) != search.training_digest:
        raise ValueError("Поиск выполнен на другом обучающем ряде.")
    if tuple(reference.feature_names) != search.feature_names:
        raise ValueError("Исходная модель и поиск должны использовать одинаковые признаки.")
    if any(reference.model.get_params().get(name) != value for name, value in search.base_parameters.items()):
        raise ValueError("Исходная модель не соответствует базовым параметрам поиска.")
    selected = run_boosting_experiment(train, validation, search.best_parameters,
                                       reference.predictions, interval_minutes)
    predictions = reference.predictions.copy()
    predictions[TUNED_METHOD] = selected.predictions[BOOSTING_METHOD]
    comparison = pd.concat([reference.comparison, selected.comparison.loc[[BOOSTING_METHOD]].rename(
        index={BOOSTING_METHOD: TUNED_METHOD})]).sort_values("mae_kw", kind="stable")
    daily = reference.daily_mae.copy()
    daily[TUNED_METHOD] = selected.daily_mae[BOOSTING_METHOD]
    return TunedBoostingExperiment(search, selected, comparison, predictions, daily)


# Сохраняет состав поиска, параметры и ошибки каждого временного блока для проверки.
def save_boosting_search(search: BoostingSearch, output_dir) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    search.summary.to_csv(path / "cv_summary.csv")
    search.fold_scores.to_csv(path / "cv_fold_scores.csv", index=False)
    search.folds.to_csv(path / "cv_windows.csv")
    search.oof_predictions.to_csv(path / "cv_predictions.csv")
    (path / "selected_parameters.json").write_text(json.dumps({
        "candidate": search.best_candidate_id, "parameters": search.best_parameters,
        "training_digest": search.training_digest,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


# Экспортирует итоговое сравнение, сохраняя исходные файлы эксперимента 3.
def save_tuned_boosting_results(experiment: TunedBoostingExperiment, output_dir) -> None:
    path = Path(output_dir)
    save_boosting_search(experiment.search, path)
    experiment.comparison.to_csv(path / "validation_metrics.csv")
    experiment.predictions.to_csv(path / "validation_predictions.csv")
    experiment.daily_mae.to_csv(path / "validation_daily_mae.csv")
