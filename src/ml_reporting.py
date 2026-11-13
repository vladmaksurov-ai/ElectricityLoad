"""Графики и обновляемые пояснения к экспериментам суточного прогноза."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ml_baselines import METHOD_LABELS, METRIC_LABELS
from .ml_boosting import BOOSTING_LABEL, BOOSTING_METHOD, BoostingExperiment
from .visualizations import PRIMARY_COLOR, SECONDARY_COLOR, SUPPORT_COLOR


METHOD_COLORS = {
    "last_value": "#727B87", "previous_day": SECONDARY_COLOR,
    "previous_week": SUPPORT_COLOR,
}


# Показывает первый день и знаковую ошибку без изменения рядов факта и прогноза.
def plot_day_forecast(day_forecast: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             height_ratios=[2, 1], constrained_layout=True)
    axes[0].plot(day_forecast.index, day_forecast["actual_kw"],
                 label="Факт", color=PRIMARY_COLOR, linewidth=2)
    axes[0].plot(day_forecast.index, day_forecast["previous_day"],
                 label="Как вчера", color=SECONDARY_COLOR, linestyle="--", linewidth=2)
    day = day_forecast.index[0]
    axes[0].set(title=f"Первые сутки валидации: {day:%d.%m.%Y}", ylabel="Мощность, кВт")
    axes[0].legend()
    error = day_forecast["previous_day"] - day_forecast["actual_kw"]
    axes[1].plot(error.index, error, color=SECONDARY_COLOR)
    axes[1].axhline(0, color="#727B87", linewidth=1)
    axes[1].set(xlabel="Начало 15-минутного интервала, местное время",
                ylabel="Прогноз − факт, кВт")
    axes[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].set_xlim(day, day_forecast.index[-1])
    for ax in axes:
        ax.grid(axis="y", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    return fig, axes


# Разносит ошибки в кВт и процентах по отдельным осям с общей сортировкой по MAE.
def plot_baseline_comparison(metrics: pd.DataFrame, start, end, *, method_labels=None, method_colors=None):
    method_labels = METHOD_LABELS if method_labels is None else method_labels
    method_colors = METHOD_COLORS if method_colors is None else method_colors
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    positions = np.arange(len(metrics))
    for ax, (metric, label) in zip(axes, METRIC_LABELS.items()):
        values = metrics[metric]
        bars = ax.barh(positions, values, color=[method_colors[m] for m in metrics.index])
        ax.bar_label(bars, labels=[f"{v:,.2f}" if pd.notna(v) else "не определён"
                                   for v in values], padding=5)
        ax.set_xlabel(label)
        ax.set_xlim(0, values.max() * 1.35 if values.notna().any() and values.max() > 0 else 1)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(positions, [method_labels[m] for m in metrics.index])
    axes[0].invert_yaxis()
    fig.suptitle(f"Валидация {start:%d.%m.%Y}–{end:%d.%m.%Y}: меньше — лучше")
    return fig, axes


# Привязывает выводы первого опыта к рассчитанным метрикам и наибольшему расхождению.
def format_day_conclusions(day_forecast: pd.DataFrame, metrics: pd.Series) -> str:
    error = day_forecast["previous_day"] - day_forecast["actual_kw"]
    worst_time = error.abs().idxmax()
    worst = day_forecast.loc[worst_time]
    bias = error.mean()
    direction = "завышает" if bias > 0 else "занижает" if bias < 0 else "не смещает"
    wape = f"{metrics['wape_pct']:.2f}%" if pd.notna(metrics["wape_pct"]) else "не определён"
    return (
        f"**Вывод эксперимента 1.** Для {day_forecast.index[0]:%d.%m.%Y} "
        f"на {len(day_forecast)} интервалах: MAE **{metrics['mae_kw']:,.2f} кВт**, "
        f"RMSE **{metrics['rmse_kw']:,.2f} кВт**, WAPE **{wape}**. "
        f"В среднем прогноз {direction} нагрузку; средняя знаковая ошибка "
        f"(прогноз − факт) **{bias:+,.2f} кВт**.\n\n"
        f"Самое большое расхождение — в **{worst_time:%H:%M}**: факт "
        f"{worst['actual_kw']:,.2f} кВт, прогноз {worst['previous_day']:,.2f} кВт, "
        f"абсолютная ошибка **{abs(error.loc[worst_time]):,.2f} кВт**. "
        "Повтор предыдущих суток не учитывает изменение уровня и формы нагрузки "
        "между днями. График ниже нуля во втором поле означает занижение прогноза, "
        "выше нуля — завышение.\n\n"
        "Это первый полный день валидации, выбранный без поиска удачного примера. "
        "Один день не представляет разные дни недели, сезоны и редкие пики; "
        "по нему нельзя выбрать лучший метод для всего периода."
    )


# Объясняет выбор по MAE и отдельно интерпретирует RMSE и общий WAPE на валидации.
def format_validation_conclusions(metrics: pd.DataFrame, forecasts: pd.DataFrame) -> str:
    best = metrics["mae_kw"].idxmin()
    best_rmse = metrics["rmse_kw"].idxmin()
    best_wape = metrics["wape_pct"].idxmin() if metrics["wape_pct"].notna().any() else None
    days = forecasts.index.normalize().nunique()
    lines = [
        f"**Вывод эксперимента 2.** Все методы оценены на одинаковых "
        f"**{len(forecasts):,} интервалах ({days} полных суток)**, "
        f"{forecasts.index[0]:%d.%m.%Y}–{forecasts.index[-1]:%d.%m.%Y}. "
        f"По основному критерию MAE лучший базовый метод — **«{METHOD_LABELS[best]}»**, "
        f"его средняя абсолютная ошибка **{metrics.loc[best, 'mae_kw']:,.2f} кВт**."
    ]
    for method in metrics.index:
        reference = metrics.loc[method, "mae_kw"]
        if method != best and reference > 0:
            improvement = (reference - metrics.loc[best, "mae_kw"]) / reference * 100
            lines.append(f"MAE ниже метода «{METHOD_LABELS[method]}» на **{improvement:.2f}%**.")
    lines.append(
        f"**RMSE** сильнее штрафует крупные промахи. Минимум у метода "
        f"«{METHOD_LABELS[best_rmse]}»: **{metrics.loc[best_rmse, 'rmse_kw']:,.2f} кВт**. "
        f"Для выбранного по MAE ориентира RMSE равен "
        f"**{metrics.loc[best, 'rmse_kw']:,.2f} кВт**; крупные ошибки остаются "
        "важны для планирования мощности."
    )
    if best_wape is not None:
        lines.append(
            f"**WAPE** выбранного метода — **{metrics.loc[best, 'wape_pct']:.2f}%**. "
            "Это сумма абсолютных ошибок, делённая на сумму абсолютного факта "
            "по всем интервалам, а не средний суточный процент. "
            "При одинаковом факте WAPE и MAE дают одинаковое ранжирование методов."
        )
    else:
        lines.append("**WAPE не определён:** сумма абсолютных фактических значений равна нулю.")
    lines.append(
        f"**Ориентир для следующего шага — «{METHOD_LABELS[best]}».** "
        "ML-модель сравниваем с ним на той же валидации, с теми же "
        "ежедневными точками старта и горизонтом 96 интервалов; основной критерий — MAE, "
        "дополнительно RMSE и WAPE. Тест июля–декабря не участвовал в оценке "
        "и выборе метода. Результат января–июня сам по себе не гарантирует "
        "такое же качество в другом сезоне."
    )
    return "\n\n".join(lines)


# Переиспользует общий график метрик для сравнения трёх ориентиров и первой ML-модели.
def plot_boosting_comparison(experiment: BoostingExperiment):
    return plot_baseline_comparison(
        experiment.comparison, experiment.predictions.index[0], experiment.predictions.index[-1],
        method_labels={**METHOD_LABELS, BOOSTING_METHOD: BOOSTING_LABEL},
        method_colors={**METHOD_COLORS, BOOSTING_METHOD: PRIMARY_COLOR},
    )


# Показывает, как ежедневная ошибка модели соотносится с лучшим базовым методом.
def plot_boosting_daily_errors(experiment: BoostingExperiment):
    baseline = experiment.comparison.loc[list(METHOD_LABELS), "mae_kw"].idxmin()
    daily = experiment.daily_mae
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(daily.index, daily[baseline], label=METHOD_LABELS[baseline],
            color=METHOD_COLORS[baseline], linestyle="--", linewidth=1.4)
    ax.plot(daily.index, daily[BOOSTING_METHOD], label=BOOSTING_LABEL,
            color=PRIMARY_COLOR, linewidth=1.5)
    ax.set(title=f"Суточный MAE: {daily.index[0]:%d.%m.%Y}–{daily.index[-1]:%d.%m.%Y}",
           xlabel="Дата прогноза; каждый день содержит 96 интервалов", ylabel="MAE, кВт", ylim=(0, None))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    ax.grid(axis="y", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    return fig, ax


# Формирует честный итог третьего опыта по выполненным расчётам, включая ухудшение качества.
def format_boosting_conclusions(experiment: BoostingExperiment) -> str:
    comparison = experiment.comparison
    baseline = comparison.loc[list(METHOD_LABELS), "mae_kw"].idxmin()
    model_metrics, baseline_metrics = comparison.loc[BOOSTING_METHOD], comparison.loc[baseline]
    labels = {**METHOD_LABELS, BOOSTING_METHOD: BOOSTING_LABEL}
    best = comparison["mae_kw"].idxmin()
    difference = baseline_metrics.mae_kw - model_metrics.mae_kw
    if baseline_metrics.mae_kw > 0:
        direction = "ниже" if difference > 0 else "выше" if difference < 0 else "такой же"
        change = f"{direction} на {abs(difference) / baseline_metrics.mae_kw * 100:.2f}%"
    else:
        change = "также нулевой" if model_metrics.mae_kw == 0 else "выше нулевой ошибки базового метода"
    summary = experiment.training_summary
    wins = int((experiment.daily_mae[BOOSTING_METHOD] < experiment.daily_mae[baseline]).sum())
    wape = f"{model_metrics.wape_pct:.2f}%" if pd.notna(model_metrics.wape_pct) else "не определён"
    return "\n\n".join([
        f"**Вывод эксперимента 3.** {BOOSTING_LABEL} обучен один раз на "
        f"**{summary['Обучающих интервалов']:,} интервалах**, "
        f"{summary['Начало обучающих целей']:%d.%m.%Y}–{summary['Конец обучающих целей']:%d.%m.%Y}, "
        f"использовано **{len(experiment.feature_names)} признаков**. "
        "Первые семь суток обучения нужны для истории. Веса модели на валидации не обновлялись; "
        "обновлялись только признаки по завершившимся дням.",
        f"**MAE модели — {model_metrics.mae_kw:,.2f} кВт**; у лучшего базового метода "
        f"«{METHOD_LABELS[baseline]}» — **{baseline_metrics.mae_kw:,.2f} кВт**. "
        f"MAE модели {change}. По главному критерию сейчас лучший результат "
        f"у **«{labels[best]}»**.",
        f"**RMSE модели — {model_metrics.rmse_kw:,.2f} кВт**, "
        f"у базового метода — {baseline_metrics.rmse_kw:,.2f} кВт. "
        "RMSE сильнее учитывает крупные промахи, поэтому его оцениваем отдельно от MAE. "
        f"**WAPE модели — {wape}**: отношение сумм ошибок и абсолютного факта "
        "по всем интервалам, без усреднения суточных процентов.",
        f"Модель дала меньший суточный MAE в **{wins} из {len(experiment.daily_mae)} дней**. "
        "Суточный график помогает увидеть периоды расхождений; победителя выбираем "
        "по общей ошибке на всех одинаковых интервалах.",
        "Это один заранее заданный набор признаков и параметров без поиска по валидации. "
        "Ранняя остановка со случайным разделением отключена. Тест июля–декабря не "
        "использовался ни при обучении, ни при сравнении. Следующий шаг — разобрать "
        "периоды крупных ошибок и проверить обоснованное изменение признаков или модели "
        "на том же протоколе; тест остаётся для итоговой проверки.",
    ])
