"""Визуализации для EDA энергопотребления."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


PRIMARY_COLOR = "#245A81"
SECONDARY_COLOR = "#9A6B13"
COMPARISON_COLOR = "#A14B5A"
SUPPORT_COLOR = "#7A8B63"
GRID_COLOR = "#D8DEE4"


def configure_eda_plot_style() -> None:
    """Задаёт единую светлую тему независимо от настроек текущего ядра."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#27313F",
            "axes.labelcolor": "#27313F",
            "axes.edgecolor": "#657180",
            "xtick.color": "#27313F",
            "ytick.color": "#27313F",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "axes.grid": False,
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "axes.prop_cycle": plt.cycler(
                color=[PRIMARY_COLOR, SECONDARY_COLOR, COMPARISON_COLOR, SUPPORT_COLOR]
            ),
        }
    )


def _style_time_series_axis(ax: Axes) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_monthly_activity_coverage(
    monthly_active_clients: pd.Series,
    total_clients: int,
) -> tuple[Figure, Axes]:
    """Строит месячный охват наблюдаемых периодов клиентов."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(
        monthly_active_clients.index,
        monthly_active_clients,
        where="post",
        color=PRIMARY_COLOR,
        linewidth=2,
    )
    ax.set(
        title="Клиенты с наблюдаемым периодом активности",
        xlabel="Месяц",
        ylabel="Количество клиентов",
    )
    ax.set_ylim(0, total_clients * 1.05)
    _style_time_series_axis(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    return fig, ax


def plot_load_resolution_comparison(
    load_15min: pd.Series,
    hourly_mean_load: pd.Series,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Сопоставляет 15-минутную и среднюю часовую мощность."""
    fig, axes_array = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
    )
    axes = (axes_array[0], axes_array[1])

    axes[0].plot(load_15min.index, load_15min, color=PRIMARY_COLOR, linewidth=1.0)
    axes[1].plot(
        hourly_mean_load.index,
        hourly_mean_load,
        color=SECONDARY_COLOR,
        linewidth=1.6,
    )

    titles = (
        "Общая мощность: 15-минутный шаг",
        "Средняя общая мощность: часовой шаг",
    )
    for ax, title in zip(axes, titles):
        ax.set(title=title, ylabel="кВт")
        _style_time_series_axis(ax)

    axes[1].set_xlabel("Дата, январь 2011")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    return fig, axes


def _plot_mean_load(
    summary: pd.DataFrame,
    *,
    title: str,
    color: str,
    linewidth: float,
    marker: str | None = None,
    markersize: float | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        summary.index,
        summary["mean_power_kw"],
        color=color,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
    )
    ax.set(title=title, xlabel="Год", ylabel="кВт")
    _style_time_series_axis(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    return fig, ax


def plot_daily_mean_load(daily_summary: pd.DataFrame) -> tuple[Figure, Axes]:
    """Строит среднюю совокупную мощность по дням."""
    return _plot_mean_load(
        daily_summary,
        title="Средняя общая мощность по дням",
        color=PRIMARY_COLOR,
        linewidth=0.9,
    )


def plot_monthly_mean_load(monthly_summary: pd.DataFrame) -> tuple[Figure, Axes]:
    """Строит среднюю совокупную мощность по месяцам."""
    return _plot_mean_load(
        monthly_summary,
        title="Средняя общая мощность по месяцам",
        color=SECONDARY_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=3,
    )


def _set_time_slot_axis(ax: Axes) -> None:
    ax.set_xticks(
        range(0, 25, 3),
        [f"{hour:02d}:00" for hour in range(0, 25, 3)],
    )
    ax.set_xlim(0, 24)


def plot_typical_daily_profile(
    daily_profile: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Строит медианный суточный профиль с межквартильным диапазоном."""
    hours = daily_profile.index.to_numpy() / 4
    median = daily_profile["median_power_kw"].to_numpy()
    q25 = daily_profile["q25_power_kw"].to_numpy()
    q75 = daily_profile["q75_power_kw"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(
        hours,
        q25,
        q75,
        color=PRIMARY_COLOR,
        alpha=0.23,
        label="25–75-й процентили",
    )
    ax.plot(
        hours,
        median,
        color=PRIMARY_COLOR,
        linewidth=2,
        label="Медианная мощность",
    )
    ax.set(
        title="Типичный суточный профиль совокупной нагрузки",
        xlabel="Время суток",
        ylabel="Мощность, кВт",
    )
    _set_time_slot_axis(ax)
    _style_time_series_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    return fig, ax


def plot_weekly_load_heatmap(
    weekly_heatmap: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Строит тепловую карту медианной мощности по дням недели и часам."""
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(
        weekly_heatmap.to_numpy(),
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
    )
    ax.set(
        title="Медианная совокупная мощность по дням недели и часам",
        xlabel="Час суток",
        ylabel="День недели",
    )
    ax.set_xticks(range(0, 24, 2), [f"{hour:02d}:00" for hour in range(0, 24, 2)])
    weekday_labels = {
        "Monday": "Понедельник", "Tuesday": "Вторник", "Wednesday": "Среда",
        "Thursday": "Четверг", "Friday": "Пятница", "Saturday": "Суббота",
        "Sunday": "Воскресенье",
    }
    ax.set_yticks(range(len(weekly_heatmap.index)),
                  [weekday_labels.get(day, day) for day in weekly_heatmap.index])
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Медианная мощность, кВт")
    fig.tight_layout()
    return fig, ax


def plot_yearly_normalized_daily_profiles(
    normalized_profiles: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Сравнивает нормированную форму суточного профиля по годам."""
    fig, ax = plt.subplots(figsize=(11, 4))
    hours = normalized_profiles.index.to_numpy() / 4
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, COMPARISON_COLOR, SUPPORT_COLOR]
    line_styles = ["-", "--", "-.", ":"]
    for color, line_style, year in zip(colors, line_styles, normalized_profiles.columns):
        ax.plot(
            hours,
            normalized_profiles[year],
            linewidth=1.8,
            linestyle=line_style,
            color=color,
            label=str(year),
        )
    ax.axhline(1, color="#6B7280", linewidth=0.8, linestyle="--")
    ax.set(
        title="Форма суточного профиля по годам",
        xlabel="Время суток",
        ylabel="Отношение к среднему\nмедианного профиля года",
    )
    _set_time_slot_axis(ax)
    _style_time_series_axis(ax)
    ax.legend(title="Год", frameon=False, ncol=4)
    return fig, ax


def plot_day_type_profiles(
    day_type_profiles: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Сопоставляет полные медианные профили будней и выходных."""
    fig, ax = plt.subplots(figsize=(11, 4))
    hours = day_type_profiles.index.to_numpy() / 4
    for column, label, color, line_style in (
        ("median_будни_kw", "Будни", PRIMARY_COLOR, "-"),
        ("median_выходные_kw", "Выходные", COMPARISON_COLOR, "--"),
    ):
        ax.plot(
            hours,
            day_type_profiles[column],
            linewidth=2,
            color=color,
            linestyle=line_style,
            label=label,
        )
    ax.set(
        title="Типичный профиль нагрузки: будни и выходные",
        xlabel="Время суток",
        ylabel="Медианная мощность, кВт",
    )
    _set_time_slot_axis(ax)
    _style_time_series_axis(ax)
    ax.legend(frameon=False)
    return fig, ax


def plot_daily_profile_with_peaks(
    daily_profile: pd.DataFrame,
    peak_summary: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Отмечает найденные оконные максимумы на медианном профиле."""
    fig, ax = plt.subplots(figsize=(11, 4))
    hours = daily_profile.index.to_numpy() / 4
    median = daily_profile["median_power_kw"]
    ax.plot(hours, median, color=PRIMARY_COLOR, linewidth=2)

    colors = [SECONDARY_COLOR, COMPARISON_COLOR]
    for color, (period, row) in zip(colors, peak_summary.iterrows()):
        x_value = row["time_slot"] / 4
        y_value = row["median_power_kw"]
        is_evening = "Вечер" in period
        ax.scatter(x_value, y_value, s=65, color=color, zorder=3)
        ax.annotate(
            f"{period}: {row['time']}",
            (x_value, y_value),
            xytext=(0, -18 if is_evening else 12),
            textcoords="offset points",
            ha="center",
            va="top" if is_evening else "bottom",
            color=color,
        )
    ax.set(
        title="Типичный суточный профиль и максимумы заданных окон",
        xlabel="Время суток",
        ylabel="Медианная мощность, кВт",
    )
    _set_time_slot_axis(ax)
    _style_time_series_axis(ax)
    fig.tight_layout()
    return fig, ax


def plot_peak_event_timing(
    timing: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Показывает число сильных событий по часу и типу дня."""
    fig, ax = plt.subplots(figsize=(11, 4))
    x_values = np.arange(24)
    width = 0.38
    ax.bar(
        x_values - width / 2,
        timing["Будни"],
        width,
        color=PRIMARY_COLOR,
        label="Будни",
    )
    ax.bar(
        x_values + width / 2,
        timing["Выходные"],
        width,
        color=COMPARISON_COLOR,
        label="Выходные",
    )
    ax.set(
        title="Распределение сильных событий по часу пика",
        xlabel="Час пика события",
        ylabel="Количество событий",
    )
    ax.set_xticks(range(0, 24, 2), [f"{hour:02d}" for hour in range(0, 24, 2)])
    _style_time_series_axis(ax)
    ax.legend(frameon=False)
    return fig, ax


def plot_client_instability(
    client_summary: pd.DataFrame,
    instability: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Сопоставляет объём клиента и относительный дневной разброс."""
    plot_data = client_summary[["client_total_energy_kwh"]].join(
        instability[["relative_daily_mad_pct"]],
        how="inner",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        plot_data["client_total_energy_kwh"],
        plot_data["relative_daily_mad_pct"],
        s=24,
        color=PRIMARY_COLOR,
        alpha=0.8,
    )
    offsets = [(5, 7), (5, 7), (5, -12), (5, 7), (5, -12)]
    top_clients = plot_data["relative_daily_mad_pct"].nlargest(5).index
    for client, offset in zip(top_clients, offsets):
        row = plot_data.loc[client]
        ax.annotate(
            client,
            (row["client_total_energy_kwh"], row["relative_daily_mad_pct"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, plot_data["relative_daily_mad_pct"].max() * 1.15)
    ax.set(
        title="Суммарная энергия и изменчивость суточного потребления",
        xlabel="Суммарная энергия клиента, кВт·ч (логарифмическая шкала)",
        ylabel="Медианное относительное отклонение, %",
    )
    _style_time_series_axis(ax)
    return fig, ax


def plot_client_share_pareto(
    client_summary: pd.DataFrame,
    top_n: int = 20,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Показывает доли крупнейших клиентов и накопленную концентрацию."""
    ranked = client_summary.sort_values("share_pct", ascending=False).head(top_n)
    fig, ax_share = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(ranked))
    ax_share.bar(positions, ranked["share_pct"], color=PRIMARY_COLOR)
    ax_share.set(
        title=f"Доля крупнейших {top_n} клиентов в общей энергии",
        xlabel="Клиент",
        ylabel="Доля общей энергии, %",
    )
    ax_share.set_xticks(positions, ranked.index, rotation=60, ha="right")
    _style_time_series_axis(ax_share)

    ax_cumulative = ax_share.twinx()
    ax_cumulative.plot(
        positions,
        ranked["cumulative_share_pct"],
        color=SECONDARY_COLOR,
        marker="o",
        markersize=3,
        linewidth=1.8,
    )
    ax_cumulative.set_ylabel("Накопленная доля, %", color=SECONDARY_COLOR)
    ax_cumulative.set_ylim(0, 100)
    ax_cumulative.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_cumulative.spines["top"].set_visible(False)
    fig.tight_layout()
    return fig, (ax_share, ax_cumulative)


def plot_relative_spike_ranking(
    spike_ranking: pd.DataFrame,
    top_n: int = 10,
) -> tuple[Figure, Axes]:
    """Строит рейтинг относительного 99-го процентиля скачков."""
    ranked = spike_ranking.nlargest(top_n, "p99_relative_change_pct").sort_values(
        "p99_relative_change_pct"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ranked.index, ranked["p99_relative_change_pct"], color=PRIMARY_COLOR)
    ax.set(
        title="Крупнейшие относительные изменения нагрузки клиентов",
        xlabel="99-й процентиль изменения относительно медианы клиента, %",
        ylabel="Клиент",
    )
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def plot_longest_zero_periods(
    zero_summary: pd.DataFrame,
    top_n: int = 10,
) -> tuple[Figure, Axes]:
    """Показывает длительности точками: длина столбца на log-шкале неоднозначна."""
    ranked = zero_summary.nlargest(top_n, "longest_zero_hours").sort_values(
        "longest_zero_hours"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(ranked))
    ax.scatter(
        ranked["longest_zero_hours"],
        positions,
        s=60,
        color=COMPARISON_COLOR,
        zorder=3,
    )
    ax.set_yticks(positions, ranked.index.astype(str))
    ax.set_xscale("log")
    for position, value in enumerate(ranked["longest_zero_hours"]):
        ax.annotate(
            f"{value:g}", (value, position), xytext=(7, 0),
            textcoords="offset points", va="center", fontsize=9,
        )
    ax.set_xlim(ranked["longest_zero_hours"].min() / 1.7,
                ranked["longest_zero_hours"].max() * 2.5)
    ax.margins(y=0.08)
    ax.set(
        title="Самые длинные нулевые периоды внутри наблюдаемой активности",
        xlabel="Длительность, часы (логарифмическая шкала)",
        ylabel="Клиент",
    )
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax

def plot_clock_change_examples(
    data: pd.DataFrame,
    dates: tuple[str, ...] = ("2014-03-30", "2014-10-26"),
) -> tuple[Figure, np.ndarray]:
    """Показывает исходные значения около окна и те же слоты соседних дат."""
    total_power = data.sum(axis=1)
    fig, axes = plt.subplots(len(dates), 1, figsize=(11, 3.5 * len(dates)),
                             constrained_layout=True, squeeze=False)
    for ax, value in zip(axes.ravel(), dates):
        date = pd.Timestamp(value)
        for offset, color, line_style in (
            (-1, "#7E8792", "--"), (1, SECONDARY_COLOR, ":"), (0, PRIMARY_COLOR, "-"),
        ):
            day = date + pd.Timedelta(days=offset)
            series = total_power.loc[day:day + pd.Timedelta(hours=4)]
            hours = series.index.hour + series.index.minute / 60
            ax.plot(hours, series, color=color, linestyle=line_style,
                    linewidth=2 if offset == 0 else 1.4,
                    label=day.strftime("%d.%m.%Y"))
        ax.axvspan(1, 2, color=COMPARISON_COLOR, alpha=0.12,
                   label="Потенциальное окно перевода часов")
        ax.set(title=f"{date:%d.%m.%Y}: сравнение с соседними датами",
               xlabel="Время по исходной метке", ylabel="Мощность, кВт",
               xlim=(0, 4), xticks=range(5),
               xticklabels=[f"{hour:02d}:00" for hour in range(5)])
        _style_time_series_axis(ax)
        ax.legend(loc="upper right", ncol=2)
    return fig, axes.ravel()


def plot_spike_case_contexts(
    active_data: pd.DataFrame,
    events: pd.DataFrame,
    padding_hours: float = 2,
) -> tuple[Figure, np.ndarray]:
    """Показывает контекст выбранных скачков без изменения значений."""
    fig, axes = plt.subplots(len(events), 1, figsize=(11, 3.2 * len(events)),
                             constrained_layout=True, squeeze=False)
    padding = pd.Timedelta(hours=padding_hours)
    for ax, ((client, timestamp), row) in zip(axes.ravel(), events.iterrows()):
        series = active_data.loc[timestamp - padding:timestamp + padding, client]
        ax.plot(series.index, series, marker="o", markersize=3,
                color=PRIMARY_COLOR, linewidth=1.5)
        ax.scatter(timestamp, row["current_kw"], color=COMPARISON_COLOR,
                   s=55, zorder=3, label="Проверяемый интервал")
        ax.axvline(timestamp, color="#7E8792", linestyle="--", linewidth=1)
        ax.set(title=f"{client} · {timestamp:%d.%m.%Y %H:%M} · исходные метки",
               xlabel="Время", ylabel="Мощность, кВт")
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        _style_time_series_axis(ax)
        ax.legend(loc="lower right")
    return fig, axes.ravel()


def plot_zero_period_context(
    active_data: pd.DataFrame,
    period: pd.Series,
    padding_days: int = 7,
) -> tuple[Figure, Axes]:
    """Показывает дневную мощность до, внутри и после внутренней нулевой серии."""
    client = period["client"]
    padding = pd.Timedelta(days=padding_days)
    series = active_data.loc[period["start"] - padding:period["last_zero"] + padding, client].copy()
    series.index -= pd.Timedelta(minutes=15)
    daily_mean = series.resample("1D").mean()
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(daily_mean.index, daily_mean, color=PRIMARY_COLOR, linewidth=1.5,
            label="Средняя дневная мощность")
    ax.axvspan(period["start"] - pd.Timedelta(minutes=15), period["last_zero"],
               color=COMPARISON_COLOR, alpha=0.16, label="Нулевая серия")
    ax.set(title=f"{client}: контекст самой длинной внутренней нулевой серии",
           xlabel="Дата по началу интервала", ylabel="Мощность, кВт")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    ax.set_ylim(bottom=0)
    _style_time_series_axis(ax)
    ax.legend(loc="upper right")
    return fig, ax
