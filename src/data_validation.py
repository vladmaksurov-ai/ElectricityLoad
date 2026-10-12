"""Проверки структуры и качества исходного датасета энергопотребления."""

from typing import Any

import numpy as np
import pandas as pd


# Собирает основные характеристики таблицы без вывода всех клиентских столбцов.
def get_dataset_overview(
    data: pd.DataFrame,
    preview_clients: int = 10,
) -> dict[str, Any]:
    """Печатает и возвращает структуру исходного датасета."""
    rows, columns = data.shape

    overview = {
        "rows": rows,
        "columns": columns,
        "start_date": data.index.min(),
        "end_date": data.index.max(),
        "first_clients": data.columns[:preview_clients].tolist(),
        "type_counts": data.dtypes.value_counts(),
    }

    print("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("-" * 45)
    print(f"Строки (временные отметки): {overview['rows']:,}")
    print(f"Столбцы (клиенты / точки учёта): {overview['columns']}")
    print(f"Начало наблюдений: {overview['start_date']}")
    print(f"Конец наблюдений:  {overview['end_date']}")
    print(f"Первые {preview_clients} клиентов: {overview['first_clients']}")
    print()
    print("Количество столбцов по типам данных:")
    print(overview["type_counts"])

    return overview


# Проверяет временную сетку, дубликаты и явные пропуски во всём датасете.
def get_time_index_quality(
    data: pd.DataFrame,
    common_intervals_count: int = 5,
) -> dict[str, Any]:
    """Печатает и возвращает показатели целостности временного индекса."""
    # Разности соседних меток позволяют одновременно увидеть основной шаг и разрывы.
    common_intervals = (
        data.index.to_series().diff().value_counts().head(common_intervals_count)
    )

    quality = {
        "is_sorted": data.index.is_monotonic_increasing,
        "duplicate_dates": int(data.index.duplicated().sum()),
        "missing_values": int(data.isna().sum().sum()),
        "common_intervals": common_intervals,
    }

    print("ПРОВЕРКА ВРЕМЕННОГО ИНДЕКСА И ДАННЫХ")
    print("-" * 50)
    print(f"Индекс расположен по возрастанию: {quality['is_sorted']}")
    print(f"Повторяющиеся временные отметки: {quality['duplicate_dates']:,}")
    print(f"Явные пропуски NaN: {quality['missing_values']:,}")
    print()
    print("Наиболее частые интервалы между наблюдениями:")

    for interval, count in quality["common_intervals"].items():
        minutes = interval.total_seconds() / 60
        print(f"{minutes:g} минут — {count:,} раз")

    return quality


# Подсчитывает недопустимые значения и полностью нулевые строки и столбцы.
def get_value_quality(data: pd.DataFrame) -> pd.Series:
    """Проверяет числовые значения всего файла, а не только медианные профили."""
    values = data.to_numpy()
    return pd.Series(
        {
            "negative_values": int((values < 0).sum()),
            "nonfinite_values": int((~np.isfinite(values)).sum()),
            "duplicate_client_names": int(data.columns.duplicated().sum()),
            "all_zero_clients": int(data.eq(0).all(axis=0).sum()),
            "all_zero_timestamps": int(data.eq(0).all(axis=1).sum()),
        },
        name="count",
    )


# Рассчитывает долю нулевых значений отдельно для каждого клиента.
def calculate_zero_share(
    data: pd.DataFrame,
    preview_clients: int = 10,
) -> pd.Series:
    """Печатает первые доли нулей и возвращает результат для всех клиентов."""
    zero_share = data.eq(0).mean().rename("zero_share")

    print("ДОЛЯ НУЛЕВЫХ ЗНАЧЕНИЙ В ИСХОДНОМ ФАЙЛЕ")
    print("-" * 47)

    # Процентный формат применяется только к компактному выводу, не меняя значения Series.
    print(
        zero_share.head(preview_clients).to_string(
            float_format=lambda value: f"{value:.2%}",
        )
    )

    return zero_share


# Проверяет регулярную временную сетку; при необходимости требует полные календарные сутки.
def validate_regular_time_index(
    data: pd.Series | pd.DataFrame,
    interval_minutes: int = 15,
    require_complete_days: bool = False,
) -> None:
    if interval_minutes <= 0 or 1440 % interval_minutes:
        raise ValueError("Шаг должен быть положительным делителем 1440 минут.")
    index = data.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans:
        raise ValueError("Нужен непустой временной индекс без пропусков.")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Временной индекс должен быть отсортирован и уникален.")
    differences = index.to_series().diff().iloc[1:].dt.total_seconds()
    if not differences.eq(interval_minutes * 60).all():
        raise ValueError(f"Нарушена сетка {interval_minutes} минут.")
    if require_complete_days:
        expected = pd.date_range(
            index[0].normalize(),
            index[-1].normalize() + pd.DateOffset(days=1),
            freq=f"{interval_minutes}min",
            inclusive="left",
        )
        if not index.equals(expected):
            raise ValueError("Ряд должен содержать только полные календарные сутки.")
        counts = data.groupby(index.normalize()).size()
        if not counts.eq(1440 // interval_minutes).all():
            raise ValueError("Есть неполные сутки или переходы часового пояса.")


# Переиспользует диагностики EDA и останавливает подготовку ML при дефектах исходных данных.
def validate_ml_source_data(
    data: pd.DataFrame,
    interval_minutes: int = 15,
    min_intervals: int = 96,
) -> pd.Series:
    validate_regular_time_index(data, interval_minutes)
    if len(data) <= min_intervals or data.shape[1] == 0:
        raise ValueError("Недостаточно данных для суточного прогноза.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise ValueError("Клиентские столбцы должны быть числовыми.")
    time_quality = get_time_index_quality(data)
    value_quality = get_value_quality(data)
    if value_quality["duplicate_client_names"]:
        raise ValueError("Есть повторяющиеся имена клиентов.")
    if value_quality["nonfinite_values"]:
        raise ValueError("Есть NaN или бесконечные значения.")
    if value_quality["negative_values"]:
        raise ValueError("Есть отрицательная мощность.")
    return pd.Series({
        "missing_values": time_quality["missing_values"],
        "duplicate_timestamps": time_quality["duplicate_dates"],
        "zero_values": int(data.eq(0).sum().sum()),
        **value_quality.to_dict(),
    }, name="quality")
