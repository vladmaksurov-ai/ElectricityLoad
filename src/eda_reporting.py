"""Неизменённый текст динамических пояснений EDA без зависимости от IPython."""


def print_activity_overview(activity_overview):
    """Выводит исходную сводку наблюдаемых периодов клиентов."""
    print(
        "Клиентов с ненулевыми показаниями: "
        f"{activity_overview['clients_with_non_zero']} "
        f"из {activity_overview['total_clients']}"
    )
    print(
        "Первое ненулевое показание после начала файла: "
        f"{activity_overview['starts_after_data_start']}"
    )
    print(
        "Последнее ненулевое показание до конца файла: "
        f"{activity_overview['ends_before_data_end']}"
    )
    print(
        "Медианная наблюдаемая длительность: "
        f"{activity_overview['median_observed_duration']}"
    )



def format_profile_exclusions(excluded_profiles):
    """Возвращает исходные пояснения причин исключения клиентов."""
    messages = []
    for client, row in excluded_profiles.iterrows():
        messages.append(
            f"**{client}:** {row['exclusion_reason']}. "
            f"Ненулевых исходных интервалов: {int(row['nonzero_intervals']):,}; "
            f"нулевых слотов типичного профиля: {int(row['zero_slots'])}/96. "
            "Причина исключения относится к выбранным признакам, "
            "а не доказывает отсутствие потребления."
        )
    return messages



def format_k_comparison(k_comparison, candidate_labels, best_silhouette_k):
    """Формирует прежнее пояснение сравнения k."""
    return (
        f"Максимальный средний silhouette: **k={best_silhouette_k}**, "
        f"**{k_comparison.loc[best_silhouette_k, 'silhouette_mean']:.4f}**. "
        f"Размеры групп этого запуска: "
        f"**{candidate_labels[best_silhouette_k].value_counts().sort_index().tolist()}**. "
        "Полосы ошибки показывают разброс по seed, не доверительные интервалы. "
        "Выбор k требует также просмотра формы и размеров групп."
    )



def format_selected_clustering(selected_k, selected_quality, cluster_members):
    """Формирует прежнюю сводку выбранного разбиения."""
    return (
        f"Выбрано **k={selected_k}**; silhouette исходного запуска "
        f"**{selected_quality['silhouette']:.4f}**, "
        f"минимальный ARI между seed **{selected_quality['ari_min']:.4f}**. "
        f"Минимальная группа: **{int(selected_quality['smallest_cluster'])} клиентов**. "
        f"Клиентов с отрицательным индивидуальным силуэтом: "
        f"**{int(cluster_members['silhouette'].lt(0).sum())}**."
    )



def format_profile_window_note():
    """Возвращает исходное пояснение временных окон и IQR."""
    return (
        "Ночь: 00:00–06:00, утро: 06:00–12:00, день: 12:00–18:00, "
        "вечер: 18:00–24:00; правая граница не включена. "
        "При нескольких равных максимумах/минимумах таблица показывает первый слот, "
        "а не уникальный момент пика. Средняя ширина IQR описывает разброс "
        "типичных профилей внутри группы, не нестабильность клиента по дням."
    )



def format_cluster_takeaways(
    features, client_profiles, excluded_profiles, energy_coverage,
    selected_quality, cluster_summary, selected_k,
):
    """Возвращает итоговый текст кластеризации без изменения формулировок."""
    lines = [
        f"**Кластеризация завершена для выбранных признаков:** {len(features)} из "
        f"{len(client_profiles)} клиентов, 96 слотов, k={selected_k}.",
        f"Исключённых клиентов: {len(excluded_profiles)}; их доля всей энергии "
        f"{energy_coverage.loc['Исключены', 'energy_share_all_pct']:.3f}%.",
        f"Silhouette выбранного разбиения: {selected_quality['silhouette']:.4f}; "
        f"минимальный ARI между seed: {selected_quality['ari_min']:.4f}.",
    ]
    for cluster, row in cluster_summary.iterrows():
        levels = row[["night_level", "morning_level", "day_level", "evening_level"]]
        # Ведущее окно выбирается по среднему уровню, отдельно от пикового слота.
        leading_window = {
            "night_level": "ночь", "morning_level": "утро",
            "day_level": "день", "evening_level": "вечер",
        }[levels.idxmax()]
        lines.append(
            f"Кластер {cluster}: {int(row['clients'])} клиентов; наиболее высокий "
            f"средний уровень медианного профиля в окне «{leading_window}»; "
            f"первый максимальный слот {row['peak_slot']} "
            f"({row['peak_normalized']:.3f} относительно среднего профиля клиента); "
            f"доля всей энергии {row['energy_share_all_pct']:.3f}%; "
            f"средняя ширина межклиентского IQR {row['mean_slot_iqr']:.3f}."
        )
    return "\n\n".join(lines)
