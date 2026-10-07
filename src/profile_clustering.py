"""Подготовка и проверка кластеров типичных суточных профилей."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from threadpoolctl import threadpool_limits

from .eda_utils import shift_to_interval_start


def get_client_typical_profiles(
    active_data: pd.DataFrame,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Строит исходные медианные суточные профили клиентов по 96 слотам."""
    # Сначала используем ту же привязку к началу интервала, что и в EDA.
    aligned = shift_to_interval_start(active_data, interval_minutes)
    time_slot = aligned.index.hour * 4 + aligned.index.minute // 15

    # Медиана берётся по датам каждого слота; строки результата — клиенты.
    profiles = aligned.groupby(time_slot).median().T
    profiles.columns = [
        f"{slot // 4:02d}:{(slot % 4) * 15:02d}"
        for slot in profiles.columns
    ]
    profiles.index.name = "client"
    return profiles


def normalize_client_profiles(profiles):
    """Нормирует форму профиля, сохраняя прежнее исключение нулевого масштаба."""
    # Нулевое среднее не заменяем единицей: такой профиль нельзя нормировать.
    profile_scale = profiles.mean(axis=1).replace(0, np.nan)
    normalized_profiles = profiles.div(profile_scale, axis=0).dropna()
    return profile_scale, normalized_profiles


def prepare_clustering_features(active_data, profiles, normalized_profiles,
                                min_complete_days=30):
    """Применяет аудит качества и возвращает признаки и исключённых клиентов."""
    profile_quality = audit_profile_quality(
        active_data, profiles, min_complete_days=min_complete_days,
    )
    excluded_profiles = profile_quality.loc[~profile_quality["included"]].copy()
    # Сохраняем порядок клиентов из аудита: он должен совпадать с метками KMeans.
    features = normalized_profiles.loc[
        profile_quality.index[profile_quality["included"]]
    ].copy()
    validate_features(features)
    return profile_quality, excluded_profiles, features


def get_history_threshold_sensitivity(profile_quality, thresholds=(0, 7, 30, 90)):
    """Проверяет прежние условия включения при разных порогах полных дней."""
    return pd.DataFrame([
        {
            "min_complete_days": threshold,
            "included_clients": int((
                profile_quality["complete_days"].ge(threshold)
                & profile_quality["nonfinite_slots"].eq(0)
                & profile_quality["negative_slots"].eq(0)
                & profile_quality["profile_mean_kw"].gt(0)
            ).sum()),
        }
        for threshold in thresholds
    ])


def get_clustering_energy_coverage(energy_summary, cluster_members,
                                   excluded_profiles, features):
    """Сверяет объём энергии включённых и исключённых клиентов."""
    all_energy_kwh = energy_summary["client_total_energy_kwh"].sum()
    included_energy_kwh = cluster_members["client_total_energy_kwh"].sum()
    excluded_energy_kwh = energy_summary.loc[
        excluded_profiles.index, "client_total_energy_kwh"
    ].sum()
    energy_coverage = pd.DataFrame({
        "clients": [len(features), len(excluded_profiles)],
        "total_energy_kwh": [included_energy_kwh, excluded_energy_kwh],
    }, index=["Кластеризованы", "Исключены"])
    # Для обеих групп знаменатель один — энергия всех исходных клиентов.
    energy_coverage["energy_share_all_pct"] = (
        energy_coverage["total_energy_kwh"] / all_energy_kwh * 100
        if all_energy_kwh > 0 else np.nan
    )
    return all_energy_kwh, included_energy_kwh, excluded_energy_kwh, energy_coverage


def validate_clustering_result(profile_quality, cluster_members, profiles,
                               features, cluster_sizes, cluster_summary,
                               all_energy_kwh, included_energy_kwh,
                               excluded_energy_kwh):
    """Создаёт общую таблицу и выполняет исходные проверки согласованности."""
    # Левое объединение сохраняет исключённых клиентов без назначения кластера.
    client_cluster_table = profile_quality.join(cluster_members, how="left")
    client_cluster_table["cluster"] = client_cluster_table["cluster"].astype("Int64")
    assert len(client_cluster_table) == len(profiles)
    assert client_cluster_table["included"].equals(client_cluster_table["cluster"].notna())
    assert cluster_sizes.sum() == len(features)
    assert np.allclose(features.mean(axis=1), 1)

    # Проверяем сохранение энергии отдельно от согласованности назначений.
    assert np.isclose(included_energy_kwh + excluded_energy_kwh, all_energy_kwh)
    assert np.isclose(cluster_summary["total_energy_kwh"].sum(), included_energy_kwh)
    if included_energy_kwh > 0:
        assert np.isclose(cluster_summary["energy_share_included_pct"].sum(), 100)
    return client_cluster_table


def audit_profile_quality(active_data, profiles, min_complete_days=30):
    """Возвращает причины исключений, не подменяя пропуски нулями."""
    if min_complete_days < 0:
        raise ValueError("Минимальное число дней не может быть отрицательным.")
    if profiles.shape[1] != 96 or not profiles.index.is_unique:
        raise ValueError("Ожидается уникальный клиент и 96 слотов в строке.")
    if set(profiles.index) != set(active_data.columns):
        raise ValueError("Клиенты профилей и исходных рядов должны совпадать.")
    # Считаем полные дни и доступность слотов внутри наблюдаемого диапазона.
    aligned = shift_to_interval_start(active_data)
    daily_counts = aligned.resample("1D").count()
    slots = aligned.index.hour * 4 + aligned.index.minute // 15
    slot_counts = aligned.groupby(slots).count().reindex(range(96), fill_value=0)
    values = profiles.to_numpy(dtype=float)
    audit = pd.DataFrame(index=profiles.index)
    audit["observed_intervals"] = active_data.count()
    audit["nonzero_intervals"] = (active_data.notna() & active_data.ne(0)).sum()
    audit["complete_days"] = daily_counts.eq(96).sum()
    audit["min_slot_observations"] = slot_counts.min()
    audit["missing_slots"] = profiles.isna().sum(axis=1)
    audit["nonfinite_slots"] = (~np.isfinite(values)).sum(axis=1)
    audit["negative_slots"] = profiles.lt(0).sum(axis=1)
    audit["zero_slots"] = profiles.eq(0).sum(axis=1)
    audit["profile_mean_kw"] = profiles.mean(axis=1)
    audit["insufficient_history"] = audit["complete_days"].lt(min_complete_days)
    # Причины собираются независимо: один клиент может нарушать несколько условий.
    reasons = {
        "нет наблюдений": audit["observed_intervals"].eq(0),
        "пропущенные слоты": audit["missing_slots"].gt(0),
        "бесконечные значения": np.isinf(values).any(axis=1),
        "отрицательный профиль": audit["negative_slots"].gt(0),
        "нулевой медианный профиль": audit["profile_mean_kw"].eq(0),
        "недостаточно полных дней": audit["insufficient_history"],
    }
    audit["exclusion_reason"] = [
        "; ".join(name for name, mask in reasons.items() if np.asarray(mask)[i])
        for i in range(len(audit))
    ]
    audit["included"] = audit["exclusion_reason"].eq("")
    return audit


def validate_features(features):
    """Проверяет матрицу нормированных профилей перед KMeans."""
    if len(features) < 3 or features.shape[1] != 96:
        raise ValueError("Нужны как минимум три клиента и 96 слотов.")
    if not features.index.is_unique or not features.columns.is_unique:
        raise ValueError("Индексы клиентов и слотов должны быть уникальны.")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Признаки должны быть конечными и неотрицательными.")
    if not np.allclose(values.mean(axis=1), 1):
        raise ValueError("Среднее каждого нормированного профиля должно быть 1.")


def fit_profile_clusters(features, k, random_state=42, n_init=20):
    """Обучает модель и сохраняет привязку меток к клиентам."""
    validate_features(features)
    distinct_profiles = len(features.drop_duplicates())
    if not 2 <= k <= min(len(features) - 1, distinct_profiles):
        raise ValueError(
            "k должно быть меньше числа клиентов "
            "и не больше числа уникальных профилей."
        )
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    # Для маленькой матрицы ограничение потоков предотвращает лишние накладные расходы BLAS.
    with threadpool_limits(limits=1):
        predicted = model.fit_predict(features)
    labels = pd.Series(predicted, index=features.index, name="cluster")
    if labels.nunique() != k:
        raise ValueError("Модель сформировала меньше групп, чем запрошено.")
    return model, labels


def compare_cluster_counts(features, k_values=range(2, 9), seeds=(42, 17, 73)):
    """Сравнивает силуэт, размеры и согласованность разбиений при разных seed."""
    validate_features(features)
    if len(set(seeds)) < 2:
        raise ValueError("Для проверки устойчивости нужны хотя бы два разных seed.")
    rows, models, assignments = [], {}, {}
    max_k = min(len(features) - 1, len(features.drop_duplicates()))
    for k in sorted(set(k_values)):
        if not 2 <= k <= max_k:
            continue
        runs, scores = [], []
        # Меняем только seed; первый запуск сохраняется для графиков и сводок.
        for seed in seeds:
            model, labels = fit_profile_clusters(features, k, seed)
            runs.append(labels)
            scores.append(silhouette_score(features, labels))
            if seed == seeds[0]:
                models[k], assignments[k] = model, labels
        sizes = runs[0].value_counts()
        # ARI сравнивает все пары запусков, не полагаясь на номера кластеров.
        ari = [adjusted_rand_score(a, b) for a, b in combinations(runs, 2)]
        rows.append({
            "k": k,
            "silhouette": scores[0],
            "silhouette_mean": np.mean(scores),
            "silhouette_std": np.std(scores),
            "inertia": models[k].inertia_,
            "smallest_cluster": sizes.min(),
            "largest_cluster": sizes.max(),
            "smallest_cluster_pct": sizes.min() / len(features) * 100,
            "singleton_clusters": sizes.eq(1).sum(),
            "ari_mean": np.mean(ari),
            "ari_min": np.min(ari),
        })
    if not rows:
        raise ValueError("Нет допустимых k: проверьте число клиентов и уникальных профилей.")
    return pd.DataFrame(rows).set_index("k"), models, assignments


def summarize_profile_clusters(features, labels, energy_summary):
    """Сводит форму профиля, разброс и энергию с явными знаменателями долей."""
    validate_features(features)
    if not labels.index.equals(features.index):
        raise ValueError("Порядок клиентов в признаках и метках должен совпадать.")
    if not features.index.isin(energy_summary.index).all():
        raise ValueError("Для каждого включённого клиента нужна сводка энергии.")
    if not energy_summary.index.is_unique:
        raise ValueError("Сводка энергии должна содержать одну строку на клиента.")
    # Форма и межклиентский разброс считаются по нормированным профилям.
    medians = features.groupby(labels).median()
    q25 = features.groupby(labels).quantile(0.25)
    q75 = features.groupby(labels).quantile(0.75)
    silhouettes = silhouette_samples(features, labels)
    # Возвращаем медиану своей группы к каждой строке клиента перед расчётом RMSE.
    distance = np.sqrt(
        ((features - medians.loc[labels.to_numpy()].to_numpy()) ** 2).mean(axis=1)
    )
    members = pd.DataFrame({
        "cluster": labels,
        "silhouette": silhouettes,
        "distance_to_median_rmse": distance,
    }).join(energy_summary[["client_total_energy_kwh", "mean_power_kw"]])
    summary = members.groupby("cluster").agg(
        clients=("cluster", "size"),
        total_energy_kwh=("client_total_energy_kwh", "sum"),
        median_client_mean_power_kw=("mean_power_kw", "median"),
        silhouette_mean=("silhouette", "mean"),
        negative_silhouette_clients=("silhouette", lambda s: s.lt(0).sum()),
        median_distance_rmse=("distance_to_median_rmse", "median"),
    )
    all_energy = energy_summary["client_total_energy_kwh"].sum()
    eligible_energy = members["client_total_energy_kwh"].sum()
    # Две доли энергии различаются знаменателем: весь файл и включённые клиенты.
    summary["client_share_pct"] = summary["clients"] / len(features) * 100
    summary["energy_share_all_pct"] = (
        summary["total_energy_kwh"] / all_energy * 100
        if all_energy > 0 else np.nan
    )
    summary["energy_share_included_pct"] = (
        summary["total_energy_kwh"] / eligible_energy * 100
        if eligible_energy > 0 else np.nan
    )
    summary["peak_slot"] = medians.idxmax(axis=1)
    summary["peak_normalized"] = medians.max(axis=1)
    summary["trough_slot"] = medians.idxmin(axis=1)
    summary["trough_normalized"] = medians.min(axis=1)
    summary["mean_slot_iqr"] = (q75 - q25).mean(axis=1)
    hour = np.arange(96) / 4
    for name, mask in {
        "night": hour < 6,
        "morning": (hour >= 6) & (hour < 12),
        "day": (hour >= 12) & (hour < 18),
        "evening": hour >= 18,
    }.items():
        summary[f"{name}_level"] = medians.loc[:, mask].mean(axis=1)
    return summary, members, medians, q25, q75


def cluster_representatives(members, per_cluster=3):
    """Выбирает ближайшие к медиане реальные профили и самый удалённый."""
    if per_cluster < 1:
        raise ValueError("Нужно хотя бы по одному примеру на кластер.")
    rows = []
    for cluster, group in members.groupby("cluster"):
        ordered = group.sort_values("distance_to_median_rmse", kind="stable")
        for client in ordered.head(per_cluster).index:
            rows.append({
                "client": client, "cluster": cluster, "role": "близкий к медиане",
            })
        farthest = ordered.index[-1]
        if farthest not in ordered.head(per_cluster).index:
            rows.append({
                "client": farthest, "cluster": cluster, "role": "самый удалённый",
            })
    return pd.DataFrame(rows).set_index("client")
