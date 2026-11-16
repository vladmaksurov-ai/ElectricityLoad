"""Компактные таблицы, графики и обновляемый вывод эксперимента 3.1."""

import matplotlib.pyplot as plt
import pandas as pd

from .ml_baselines import METHOD_LABELS
from .ml_boosting import BOOSTING_METHOD
from .ml_reporting import METHOD_COLORS, plot_baseline_comparison
from .ml_tuning import ORIGINAL_LABEL, TUNED_LABEL, TUNED_METHOD, BoostingSearch, TunedBoostingExperiment
from .visualizations import PRIMARY_COLOR, COMPARISON_COLOR


SEARCH_COLUMNS = {
    "cv_mae_kw": "MAE CV, кВт", "cv_rmse_kw": "RMSE CV, кВт", "cv_wape_pct": "WAPE CV, %",
    "max_leaf_nodes": "Листьев", "min_samples_leaf": "Строк в листе",
    "learning_rate": "learning_rate", "max_iter": "Итераций", "l2_regularization": "L2",
}


# Показывает лучшие конфигурации и исходную модель, даже если она не попала в верхние строки.
def get_tuning_leaderboard(search: BoostingSearch, top_n: int = 8) -> pd.DataFrame:
    identifiers = search.summary.index[:top_n].tolist()
    if "original" not in identifiers:
        identifiers.append("original")
    table = search.summary.loc[identifiers, list(SEARCH_COLUMNS)].rename(columns=SEARCH_COLUMNS).round(2)
    table.insert(0, "Выбрана", [name == search.best_candidate_id for name in identifiers])
    return table.rename(index={"original": "Исходная 3"}).rename_axis("Конфигурация")


# Объясняет даты и размеры расширяющихся временных блоков.
def get_tuning_fold_table(search: BoostingSearch) -> pd.DataFrame:
    return search.folds.drop(columns="end_exclusive").rename(columns={
        "train_start": "Обучение с", "train_end": "Обучение до", "validation_start": "Проверка с",
        "validation_end": "Проверка до", "train_intervals": "Строк обучения", "intervals": "Строк проверки",
    }).rename_axis("Блок")


# Сопоставляет исходные и выбранные параметры, включая неизменные настройки протокола.
def get_tuned_parameter_table(search: BoostingSearch) -> pd.DataFrame:
    return pd.DataFrame({"Исходная модель": search.base_parameters,
                         "Выбрано по CV": search.best_parameters}).rename_axis("Параметр")


# Сверяет исходную и выбранную конфигурации на каждом из одних и тех же блоков CV.
def get_tuning_fold_comparison(search: BoostingSearch) -> pd.DataFrame:
    scores = search.fold_scores.pivot(index="fold", columns="candidate", values="mae_kw")
    result = search.folds[["validation_start", "validation_end"]].rename(columns={
        "validation_start": "Проверка с", "validation_end": "Проверка до"})
    result["MAE исходной, кВт"] = scores["original"]
    result["MAE выбранной, кВт"] = scores[search.best_candidate_id]
    return result.round(2).rename_axis("Блок")


# Ранжирует поиск только по CV, не используя ошибки внешней валидации 2014 года.
def plot_tuning_search(search: BoostingSearch, top_n: int = 8):
    identifiers = search.summary.index[:top_n].tolist()
    if "original" not in identifiers:
        identifiers.append("original")
    values = search.summary.loc[identifiers, "cv_mae_kw"]
    colors = [PRIMARY_COLOR if name == search.best_candidate_id else
              METHOD_COLORS["previous_day"] if name == "original" else "#A8B3BF" for name in identifiers]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = ax.barh(range(len(values)), values, color=colors)
    ax.bar_label(bars, labels=[f"{value:,.2f}" for value in values], padding=5)
    ax.set_yticks(range(len(values)), ["Исходная 3" if name == "original" else name for name in identifiers])
    ax.invert_yaxis()
    ax.axvline(search.summary.loc["original", "cv_mae_kw"], color=METHOD_COLORS["previous_day"],
               linestyle="--", linewidth=1, label="MAE исходной модели на CV")
    ax.set(xlabel="Общий MAE по внутренним временным блокам, кВт",
           title=f"Подбор на {search.folds.validation_start.min():%Y} году: лучшие варианты и исходная модель",
           xlim=(0, values.max() * 1.22))
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return fig, ax


# Выделяет сравнение выбранного бустинга, исходного бустинга и лучшего простого ориентира.
def get_tuned_validation_comparison(experiment: TunedBoostingExperiment) -> pd.DataFrame:
    baseline = experiment.comparison.loc[list(METHOD_LABELS), "mae_kw"].idxmin()
    return experiment.comparison.loc[[TUNED_METHOD, BOOSTING_METHOD, baseline]].sort_values("mae_kw", kind="stable")


# Переиспользует общие шкалы MAE, RMSE и WAPE для внешнего сравнения на 2014 году.
def plot_tuned_validation_comparison(experiment: TunedBoostingExperiment):
    metrics = get_tuned_validation_comparison(experiment)
    return plot_baseline_comparison(
        metrics, experiment.predictions.index[0], experiment.predictions.index[-1],
        method_labels={**METHOD_LABELS, BOOSTING_METHOD: ORIGINAL_LABEL, TUNED_METHOD: TUNED_LABEL},
        method_colors={**METHOD_COLORS, BOOSTING_METHOD: PRIMARY_COLOR, TUNED_METHOD: COMPARISON_COLOR},
    )


# Описывает относительное изменение без деления на ноль и без обещания улучшения.
def _relative_change(value, reference):
    if reference == 0:
        return "без изменения" if value == 0 else f"рост от нуля до {value:,.2f}"
    change = (reference - value) / reference * 100
    return (f"снижение на {change:.2f}%" if change > 0 else
            f"рост на {-change:.2f}%" if change < 0 else "без изменения")


# Формирует итог по выбранной CV-конфигурации, даже если она хуже исходной на 2014 году.
def format_tuning_conclusions(experiment: TunedBoostingExperiment) -> str:
    search = experiment.search
    best_cv = search.summary.loc[search.best_candidate_id]
    original_cv = search.summary.loc["original"]
    metrics = experiment.comparison
    selected = metrics.loc[TUNED_METHOD]
    original = metrics.loc[BOOSTING_METHOD]
    baseline = metrics.loc[list(METHOD_LABELS), "mae_kw"].idxmin()
    names = {**METHOD_LABELS, BOOSTING_METHOD: ORIGINAL_LABEL, TUNED_METHOD: TUNED_LABEL}
    winner = metrics.mae_kw.idxmin()
    wins = int((experiment.daily_mae[TUNED_METHOD] < experiment.daily_mae[BOOSTING_METHOD]).sum())
    parameters = ", ".join(f"{name}={search.best_parameters[name]}" for name in
                            ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"))
    wape = f"{selected.wape_pct:.2f}%" if pd.notna(selected.wape_pct) else "не определён"
    return "\n\n".join([
        f"**Вывод эксперимента 3.1.** Проверено **{len(search.summary)} конфигураций** "
        f"на **{len(search.folds)} последовательных временных блоках** внутри train "
        f"({len(search.fold_scores)} обучений). Главный критерий выбора — MAE, объединённый "
        "по всем интервалам этих блоков. Внешняя валидация 2014 года в поиске не участвовала.",
        f"Выбрана конфигурация **{search.best_candidate_id}**: `{parameters}`. "
        f"Её MAE на CV — **{best_cv.cv_mae_kw:,.2f} кВт**, исходной модели — "
        f"**{original_cv.cv_mae_kw:,.2f} кВт**; {_relative_change(best_cv.cv_mae_kw, original_cv.cv_mae_kw)}.",
        f"После обучения выбранной конфигурации на полном train на валидации "
        f"{experiment.predictions.index[0]:%d.%m.%Y}–{experiment.predictions.index[-1]:%d.%m.%Y}: "
        f"**MAE {selected.mae_kw:,.2f} кВт**, у исходной модели **{original.mae_kw:,.2f} кВт** "
        f"({_relative_change(selected.mae_kw, original.mae_kw)}). У метода "
        f"«{METHOD_LABELS[baseline]}» MAE **{metrics.loc[baseline, 'mae_kw']:,.2f} кВт**.",
        f"**RMSE выбранной модели — {selected.rmse_kw:,.2f} кВт**, исходной — "
        f"{original.rmse_kw:,.2f} кВт ({_relative_change(selected.rmse_kw, original.rmse_kw)}). "
        f"**WAPE выбранной модели — {wape}**, исходной — {original.wape_pct:.2f}%. "
        "WAPE рассчитан отношением сумм, а RMSE — корнем общей средней квадратичной ошибки.",
        f"Суточный MAE выбранной модели ниже исходной в **{wins} из {len(experiment.daily_mae)} дней**. "
        f"По общей валидационной MAE лучший результат сейчас у **«{names[winner]}»**.",
        "Это лучшая конфигурация среди проверенных вариантов по внутреннему CV, "
        "а не доказанный глобальный оптимум. Признаки и протокол прогнозирования не менялись. "
        "После просмотра внешних метрик другой вариант не подбирался. Тест июля–декабря "
        "не использовался и остаётся для итоговой проверки.",
    ])
