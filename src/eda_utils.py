"""Вспомогательные функции для EDA энергопотребления."""

from typing import Any

import numpy as np
import pandas as pd


WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def get_observed_activity(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оставляет значения между первым и последним ненулевыми интервалами."""
    non_zero = data.ne(0)
    started = non_zero.cummax(axis=0)

    # Обратный проход отмечает интервалы до последнего ненулевого значения.
    finished = non_zero.iloc[::-1].cummax(axis=0).iloc[::-1]
    activity_mask = started & finished

    return data.where(activity_mask), activity_mask


def get_activity_periods(
    active_data: pd.DataFrame,
    activity_mask: pd.DataFrame,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Рассчитывает границы и длительность наблюдаемого периода клиентов."""
    activity_periods = pd.DataFrame(
        {
            "start": active_data.apply(pd.Series.first_valid_index),
            "end": active_data.apply(pd.Series.last_valid_index),
        }
    )
    activity_periods["duration"] = (
        activity_periods["end"] - activity_periods["start"]
    )
    activity_periods["observed_intervals"] = activity_mask.sum(axis=0)
    activity_periods["observed_duration"] = (
        activity_periods["duration"] + pd.Timedelta(minutes=interval_minutes)
    )

    return activity_periods


def get_activity_overview(
    activity_periods: pd.DataFrame,
    data_index: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Собирает основные показатели охвата клиентов."""
    if data_index.empty:
        raise ValueError("Временной индекс не должен быть пустым.")

    return {
        "clients_with_non_zero": int(activity_periods["start"].notna().sum()),
        "total_clients": len(activity_periods),
        "starts_after_data_start": int(
            (activity_periods["start"] > data_index.min()).sum()
        ),
        "ends_before_data_end": int(
            (activity_periods["end"] < data_index.max()).sum()
        ),
        "median_observed_duration": activity_periods["observed_duration"].median(),
    }


def get_monthly_activity_coverage(
    activity_mask: pd.DataFrame,
    interval_minutes: int = 15,
) -> pd.Series:
    """Считает клиентов, чей наблюдаемый период пересекает каждый месяц."""
    monthly_presence = activity_mask.copy()
    monthly_presence.index = monthly_presence.index - pd.Timedelta(
        minutes=interval_minutes
    )

    return monthly_presence.resample("MS").max().sum(axis=1)


def shift_to_interval_start(
    values: pd.Series | pd.DataFrame,
    interval_minutes: int = 15,
) -> pd.Series | pd.DataFrame:
    """Переносит метки конца интервалов к началу интервалов."""
    shifted = values.copy()
    shifted.index = shifted.index - pd.Timedelta(minutes=interval_minutes)
    return shifted


def get_total_load(
    active_data: pd.DataFrame,
    interval_hours: float = 0.25,
) -> pd.DataFrame:
    """Рассчитывает общую мощность и энергию каждого исходного интервала."""
    total_power = active_data.sum(axis=1).rename("total_power_kw")
    total_energy = (total_power * interval_hours).rename("total_energy_kwh")

    return pd.concat([total_power, total_energy], axis=1)


def summarize_load(
    load: pd.Series,
    group_interval: str,
    source_interval_minutes: int = 15,
    source_interval_hours: float = 0.25,
) -> pd.DataFrame:
    """Агрегирует исходную мощность и рассчитывает энергию по периодам."""
    interval_load = shift_to_interval_start(load, source_interval_minutes)

    return (
        interval_load.resample(group_interval, closed="left", label="left")
        .agg(
            mean_power_kw="mean",
            peak_power_kw="max",
            min_power_kw="min",
            energy_kwh=lambda values: values.sum(min_count=1)
            * source_interval_hours,
        )
    )


def validate_load_aggregations(
    raw_data: pd.DataFrame,
    total_power: pd.Series,
    total_energy: pd.Series,
    hourly_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    source_interval_hours: float = 0.25,
) -> dict[str, bool]:
    """Проверяет согласованность общей мощности и энергии между масштабами."""
    checks = {
        "power_total_matches_raw_data": bool(
            np.allclose(total_power.to_numpy(), raw_data.sum(axis=1).to_numpy())
        ),
        "first_hour_energy_matches_source": bool(
            np.isclose(
                hourly_summary["energy_kwh"].iloc[0],
                total_power.iloc[:4].sum() * source_interval_hours,
            )
        ),
        "daily_energy_matches_source": bool(
            np.isclose(daily_summary["energy_kwh"].sum(), total_energy.sum())
        ),
        "monthly_energy_matches_source": bool(
            np.isclose(monthly_summary["energy_kwh"].sum(), total_energy.sum())
        ),
    }

    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise AssertionError(
            "Не пройдены проверки агрегации: " + ", ".join(failed_checks)
        )

    return checks


def add_calendar_features(
    total_load: pd.DataFrame,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Добавляет признаки календаря по началу исходного интервала."""
    result = total_load.copy()
    interval_start = result.index - pd.Timedelta(minutes=interval_minutes)

    result["hour"] = interval_start.hour
    result["minute"] = interval_start.minute
    result["weekday"] = interval_start.day_name()
    result["is_weekend"] = result["weekday"].isin(
        ["Saturday", "Sunday"]
    )
    result["day_type"] = result["is_weekend"].map(
        {False: "Будни", True: "Выходные"}
    )
    result["month"] = interval_start.month
    result["year"] = interval_start.year
    result["time_slot"] = (
        interval_start.hour * (60 // interval_minutes)
        + interval_start.minute // interval_minutes
    )

    return result


def get_daily_load_profile(total_load: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает типичную нагрузку каждого 15-минутного слота суток."""
    return (
        total_load.groupby("time_slot")
        .agg(
            median_power_kw=("total_power_kw", "median"),
            mean_power_kw=("total_power_kw", "mean"),
            q25_power_kw=("total_power_kw", lambda values: values.quantile(0.25)),
            q75_power_kw=("total_power_kw", lambda values: values.quantile(0.75)),
            peak_power_kw=("total_power_kw", "max"),
            min_power_kw=("total_power_kw", "min"),
        )
        .sort_index()
    )


def get_weekday_hour_profile(total_load: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает профиль мощности для каждого дня недели и часа."""
    return (
        total_load.groupby(["weekday", "hour"])
        .agg(
            median_power_kw=("total_power_kw", "median"),
            mean_power_kw=("total_power_kw", "mean"),
            peak_power_kw=("total_power_kw", "max"),
            min_power_kw=("total_power_kw", "min"),
        )
        .reindex(WEEKDAY_ORDER, level="weekday")
    )


def get_weekly_heatmap(weekday_hour_profile: pd.DataFrame) -> pd.DataFrame:
    """Преобразует недельный профиль в матрицу день недели × час."""
    return (
        weekday_hour_profile["median_power_kw"]
        .unstack("hour")
        .reindex(WEEKDAY_ORDER)
    )


def get_yearly_normalized_daily_profiles(
    total_load: pd.DataFrame,
) -> pd.DataFrame:
    """Сравнивает форму суточного профиля по годам без масштаба нагрузки."""
    profiles = (
        total_load.groupby(["time_slot", "year"])["total_power_kw"]
        .median()
        .unstack("year")
        .sort_index()
    )
    annual_mean = profiles.mean(axis=0).replace(0, np.nan)
    return profiles.div(annual_mean, axis="columns")


def get_day_type_profiles(
    total_load: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сравнивает полные суточные профили и общие показатели типов дней."""
    profiles = (
        total_load.groupby(["time_slot", "day_type"])["total_power_kw"]
        .agg(["mean", "median"])
        .unstack("day_type")
        .sort_index()
    )
    profiles.columns = [
        f"{stat}_{day_type.lower()}_kw"
        for stat, day_type in profiles.columns
    ]
    profiles["median_difference_kw"] = (
        profiles["median_выходные_kw"] - profiles["median_будни_kw"]
    )

    interval_start = total_load.index - pd.Timedelta(minutes=15)
    daily = pd.DataFrame(
        {
            "date": interval_start.normalize(),
            "day_type": total_load["day_type"].to_numpy(),
            "energy_kwh": total_load["total_energy_kwh"].to_numpy(),
            "power_kw": total_load["total_power_kw"].to_numpy(),
        },
        index=total_load.index,
    )
    daily_summary = (
        daily.groupby(["date", "day_type"])
        .agg(
            daily_energy_kwh=("energy_kwh", "sum"),
            daily_mean_power_kw=("power_kw", "mean"),
            daily_peak_power_kw=("power_kw", "max"),
        )
        .reset_index()
    )
    comparison = (
        daily_summary.groupby("day_type")
        .agg(
            days=("date", "nunique"),
            median_daily_energy_kwh=("daily_energy_kwh", "median"),
            mean_daily_energy_kwh=("daily_energy_kwh", "mean"),
            median_daily_peak_kw=("daily_peak_power_kw", "median"),
        )
        .reindex(["Будни", "Выходные"])
    )
    peak_slots = {
        "Будни": int(profiles["median_будни_kw"].idxmax()),
        "Выходные": int(profiles["median_выходные_kw"].idxmax()),
    }
    comparison["median_profile_peak_time"] = [
        slot_to_time(peak_slots[day_type])
        for day_type in comparison.index
    ]

    return profiles, comparison


def get_window_peak_summary(
    profile: pd.Series,
    windows: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Находит максимумы профиля в заранее заданных окнах."""
    rows = []
    for period, (first_slot, last_slot) in windows.items():
        window = profile.loc[first_slot:last_slot]
        peak_slot = int(window.idxmax())
        rows.append(
            {
                "period": period,
                "time_slot": peak_slot,
                "time": slot_to_time(peak_slot),
                "median_power_kw": float(window.loc[peak_slot]),
                "is_window_boundary": peak_slot in (first_slot, last_slot),
            }
        )

    return pd.DataFrame(rows).set_index("period")


def get_day_type_window_peaks(
    total_load: pd.DataFrame,
    windows: dict[str, tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Находит типичные пики отдельно для будней и выходных."""
    profiles = (
        total_load.groupby(["time_slot", "day_type"])["total_power_kw"]
        .median()
        .unstack("day_type")
        .reindex(columns=["Будни", "Выходные"])
    )
    summaries = []
    for day_type in profiles.columns:
        summary = get_window_peak_summary(profiles[day_type], windows).reset_index()
        summary.insert(0, "day_type", day_type)
        summaries.append(summary)

    return profiles, pd.concat(summaries, ignore_index=True)


def get_high_load_events(
    total_load: pd.DataFrame,
    quantile: float = 0.99,
    interval_minutes: int = 15,
) -> tuple[pd.DataFrame, float]:
    """Объединяет соседние интервалы выше заданного квантиля в события."""
    power = total_load["total_power_kw"]
    threshold_kw = float(power.quantile(quantile))
    is_high = power.ge(threshold_kw)
    event_id = (is_high & ~is_high.shift(fill_value=False)).cumsum()

    event_rows = total_load.loc[
        is_high,
        ["total_power_kw", "hour", "day_type", "weekday"],
    ].copy()
    event_rows["event_id"] = event_id.loc[is_high].to_numpy()
    event_rows["timestamp"] = event_rows.index

    events = event_rows.groupby("event_id").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        intervals=("timestamp", "size"),
        mean_power_kw=("total_power_kw", "mean"),
        peak_power_kw=("total_power_kw", "max"),
    )
    peak_timestamps = event_rows.groupby("event_id")["total_power_kw"].idxmax()
    events["peak_timestamp"] = peak_timestamps
    events["duration_minutes"] = events["intervals"] * interval_minutes
    events["peak_hour"] = [
        int(total_load.loc[timestamp, "hour"])
        for timestamp in events["peak_timestamp"]
    ]
    events["day_type"] = [
        total_load.loc[timestamp, "day_type"]
        for timestamp in events["peak_timestamp"]
    ]

    return events.sort_values("peak_power_kw", ascending=False), threshold_kw


def get_peak_event_timing(high_load_events: pd.DataFrame) -> pd.DataFrame:
    """Считает число сильных событий по часу и типу дня."""
    return (
        high_load_events.groupby(["peak_hour", "day_type"])
        .size()
        .unstack("day_type", fill_value=0)
        .reindex(range(24), fill_value=0)
        .reindex(columns=["Будни", "Выходные"], fill_value=0)
    )


def get_client_energy_summary(
    active_data: pd.DataFrame,
    interval_hours: float = 0.25,
) -> pd.DataFrame:
    """Рассчитывает объём, долю и базовые характеристики клиентов."""
    summary = pd.DataFrame(
        {
            "client_total_energy_kwh": active_data.sum(axis=0) * interval_hours,
            "mean_power_kw": active_data.mean(axis=0),
            "peak_power_kw": active_data.max(axis=0),
            "observed_intervals": active_data.count(axis=0),
            "zero_pct": active_data.eq(0).sum(axis=0)
            / active_data.count(axis=0)
            * 100,
        }
    )
    total_energy = summary["client_total_energy_kwh"].sum()
    summary["share_pct"] = (
        summary["client_total_energy_kwh"] / total_energy * 100
    )
    summary["cumulative_share_pct"] = (
        summary["share_pct"].sort_values(ascending=False).cumsum()
    )
    return summary.sort_values("client_total_energy_kwh", ascending=False)


def get_client_daily_instability(
    active_data: pd.DataFrame,
    interval_minutes: int = 15,
    interval_hours: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оценивает разброс энергии только по полным наблюдаемым суткам."""
    aligned = shift_to_interval_start(active_data, interval_minutes)
    daily_energy = aligned.resample("1D").sum(min_count=1) * interval_hours
    daily_counts = aligned.resample("1D").count()
    expected_intervals = int(24 * 60 / interval_minutes)
    complete_daily_energy = daily_energy.where(
        daily_counts.eq(expected_intervals)
    )

    median_energy = complete_daily_energy.median(axis=0)
    absolute_deviation = complete_daily_energy.sub(
        median_energy, axis="columns"
    ).abs()
    mad_energy = absolute_deviation.median(axis=0)
    q25 = complete_daily_energy.quantile(0.25, axis=0)
    q75 = complete_daily_energy.quantile(0.75, axis=0)
    safe_median = median_energy.where(median_energy.gt(0))

    summary = pd.DataFrame(
        {
            "complete_days": complete_daily_energy.count(axis=0),
            "median_daily_energy_kwh": median_energy,
            "daily_energy_mad_kwh": mad_energy,
            "relative_daily_mad_pct": mad_energy / safe_median * 100,
            "relative_daily_iqr_pct": (q75 - q25) / safe_median * 100,
            "zero_day_pct": complete_daily_energy.eq(0).sum(axis=0)
            / complete_daily_energy.count(axis=0)
            * 100,
        }
    )
    defined = summary.loc[
        summary["relative_daily_mad_pct"].notna()
    ].sort_values("relative_daily_mad_pct", ascending=False)
    undefined = summary.loc[
        summary["relative_daily_mad_pct"].isna()
    ].sort_values("median_daily_energy_kwh")

    return defined, undefined


def get_client_peak_contributions(
    active_data: pd.DataFrame,
    total_power: pd.Series,
    high_load_quantile: float = 0.99,
) -> pd.DataFrame:
    """Сравнивает общую долю клиента с его вкладом в сильную нагрузку."""
    total_by_client = active_data.sum(axis=0)
    overall_share = total_by_client / total_by_client.sum() * 100

    high_load_mask = total_power.ge(total_power.quantile(high_load_quantile))
    high_load_total = active_data.loc[high_load_mask].sum(axis=0)
    high_load_share = high_load_total / high_load_total.sum() * 100

    system_peak_timestamp = total_power.idxmax()
    peak_values = active_data.loc[system_peak_timestamp].fillna(0)
    peak_share = peak_values / peak_values.sum() * 100

    result = pd.DataFrame(
        {
            "overall_energy_share_pct": overall_share,
            "high_load_energy_share_pct": high_load_share,
            "system_peak_share_pct": peak_share,
        }
    )
    result["high_vs_overall_pp"] = (
        result["high_load_energy_share_pct"]
        - result["overall_energy_share_pct"]
    )
    result.attrs["system_peak_timestamp"] = system_peak_timestamp
    result.attrs["system_peak_power_kw"] = float(
        total_power.loc[system_peak_timestamp]
    )
    return result.sort_values("system_peak_share_pct", ascending=False)


def get_spike_ranking(
    active_data: pd.DataFrame,
    quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ранжирует скачки в абсолютном и относительном масштабе клиента."""
    change_kw = active_data.diff().abs()
    median_non_zero_power = active_data.mask(active_data.eq(0)).median(axis=0)
    p99_absolute = change_kw.quantile(quantile)
    p99_relative = p99_absolute / median_non_zero_power.replace(0, np.nan) * 100
    median_change = change_kw.median(axis=0)
    mad_change = change_kw.sub(median_change, axis="columns").abs().median(axis=0)
    robust_denominator = (1.4826 * mad_change).replace(0, np.nan)

    ranking = pd.DataFrame(
        {
            "p99_abs_change_kw": p99_absolute,
            "median_non_zero_power_kw": median_non_zero_power,
            "p99_relative_change_pct": p99_relative,
            "p99_robust_score": (
                p99_absolute - median_change
            ) / robust_denominator,
        }
    )
    return change_kw, ranking


def get_clock_change_windows(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Возвращает интервалы 01:00–02:00 последних воскресений марта и октября."""
    timestamps = []
    for year in sorted(index.year.unique()):
        for month in (3, 10):
            month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            last_sunday = month_end - pd.Timedelta(days=(month_end.weekday() + 1) % 7)
            timestamps.extend(
                pd.date_range(
                    last_sunday + pd.Timedelta(hours=1),
                    periods=4,
                    freq="15min",
                )
            )
    return pd.DatetimeIndex(timestamps).intersection(index)


def get_clock_change_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает потенциальные окна перевода часов с соседними датами."""
    windows = get_clock_change_windows(data.index)
    total_power = data.sum(axis=1)
    rows = []
    for date in windows.normalize().unique():
        timestamps = windows[windows.normalize() == date]
        window = data.loc[timestamps]
        # В сравнении участвуют только соседние даты в тех же исходных слотах.
        reference_times = (timestamps - pd.Timedelta(days=1)).append(
            timestamps + pd.Timedelta(days=1)
        )
        reference = total_power.reindex(reference_times).dropna()
        window_mean = float(window.sum(axis=1).mean())
        reference_mean = float(reference.mean()) if len(reference) else np.nan
        rows.append(
            {
                "date": date,
                "season": "Весна" if date.month == 3 else "Осень",
                "intervals": len(timestamps),
                "mean_power_kw": window_mean,
                "reference_mean_kw": reference_mean,
                "window_to_reference": (
                    window_mean / reference_mean if reference_mean > 0 else np.nan
                ),
                "min_nonzero_clients": int(window.ne(0).sum(axis=1).min()),
                "all_zero_intervals": int(window.eq(0).all(axis=1).sum()),
            }
        )
    columns = [
        "date", "season", "intervals", "mean_power_kw", "reference_mean_kw",
        "window_to_reference", "min_nonzero_clients", "all_zero_intervals",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("date")


def get_top_spike_events(
    active_data: pd.DataFrame,
    change_kw: pd.DataFrame,
    clients: pd.Index | list[str],
    events_per_client: int = 2,
    clock_change_windows: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Показывает соседние значения для крупнейших скачков клиентов."""
    clock_change_windows = (
        pd.DatetimeIndex([])
        if clock_change_windows is None
        else clock_change_windows
    )
    rows = []
    for client in clients:
        for timestamp, jump_kw in change_kw[client].nlargest(
            events_per_client
        ).items():
            position = active_data.index.get_loc(timestamp)
            previous_kw = active_data[client].iloc[position - 1]
            current_kw = active_data[client].iloc[position]
            next_kw = (
                active_data[client].iloc[position + 1]
                if position + 1 < len(active_data)
                else np.nan
            )
            rows.append(
                {
                    "client": client,
                    "timestamp": timestamp,
                    "previous_kw": previous_kw,
                    "current_kw": current_kw,
                    "next_kw": next_kw,
                    "jump_kw": jump_kw,
                    "touches_zero": previous_kw == 0 or current_kw == 0,
                    "clock_change_window": timestamp in clock_change_windows,
                }
            )
    return pd.DataFrame(rows).set_index(["client", "timestamp"])


def get_zero_periods(
    active_data: pd.DataFrame,
    interval_minutes: int = 15,
    systemwide_zero_times: pd.DatetimeIndex | None = None,
    clock_change_windows: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Выделяет последовательные нулевые эпизоды внутри периода клиента."""
    systemwide_zero_times = (
        pd.DatetimeIndex([])
        if systemwide_zero_times is None
        else systemwide_zero_times
    )
    clock_change_windows = (
        pd.DatetimeIndex([])
        if clock_change_windows is None
        else clock_change_windows
    )
    period_tables = []

    for client_name, client in active_data.items():
        is_zero = client.eq(0)
        run_start = is_zero & ~is_zero.shift(fill_value=False)
        run_id = run_start.cumsum()
        zero_times = client.index[is_zero]

        zero_rows = pd.DataFrame(
            {
                "time": zero_times,
                "run_id": run_id[is_zero].to_numpy(),
                "systemwide_zero": zero_times.isin(systemwide_zero_times),
                "clock_change_window": zero_times.isin(clock_change_windows),
            }
        )
        client_periods = (
            zero_rows.groupby("run_id", sort=False)
            .agg(
                start=("time", "min"),
                last_zero=("time", "max"),
                intervals=("time", "size"),
                touches_systemwide_zero=("systemwide_zero", "any"),
                touches_clock_change=("clock_change_window", "any"),
            )
            .reset_index(drop=True)
        )
        client_periods.insert(0, "client", client_name)
        client_periods["duration"] = pd.to_timedelta(
            client_periods["intervals"] * interval_minutes,
            unit="min",
        )
        period_tables.append(client_periods)

    return pd.concat(period_tables, ignore_index=True)


def get_zero_periods_summary(
    zero_periods: pd.DataFrame,
    clients: pd.Index,
    interval_minutes: int = 15,
    minimum_hours: float = 1,
) -> pd.DataFrame:
    """Суммирует длительность и повторяемость нулевых эпизодов."""
    minimum_intervals = int(minimum_hours * 60 / interval_minutes)
    summary = (
        zero_periods.groupby("client")
        .agg(
            longest_zero_intervals=("intervals", "max"),
            zero_periods_at_least_1h=(
                "intervals",
                lambda values: values.ge(minimum_intervals).sum(),
            ),
            total_zero_intervals=("intervals", "sum"),
            clock_change_related_periods=("touches_clock_change", "sum"),
        )
        .reindex(clients, fill_value=0)
    )
    summary["longest_zero_hours"] = (
        summary["longest_zero_intervals"] * interval_minutes / 60
    )
    return summary[
        [
            "longest_zero_hours",
            "zero_periods_at_least_1h",
            "total_zero_intervals",
            "clock_change_related_periods",
        ]
    ].sort_values("longest_zero_hours", ascending=False)


def get_energy_groups(
    client_summary: pd.DataFrame,
    groups: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит клиентов на квантили по суммарной энергии."""
    labels = [f"Q{number}" for number in range(1, groups + 1)]
    details = client_summary.copy()
    details["energy_group"] = pd.qcut(
        details["client_total_energy_kwh"],
        q=groups,
        labels=labels,
    )
    summary = details.groupby("energy_group", observed=True).agg(
        clients=("client_total_energy_kwh", "size"),
        sum_group_energy_kwh=("client_total_energy_kwh", "sum"),
    )
    summary["share_pct"] = (
        summary["sum_group_energy_kwh"]
        / details["client_total_energy_kwh"].sum()
        * 100
    )
    return details, summary


def slot_to_time(slot: int) -> str:
    """Преобразует номер 15-минутного слота суток в строку времени HH:MM."""
    return f"{slot // 4:02d}:{(slot % 4) * 15:02d}"
