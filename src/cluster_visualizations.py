"""Графики кластеризации: прежние шкалы, цвета и выборки из EDA."""


import matplotlib.pyplot as plt
import numpy as np


CLUSTER_COLORS = ("#4477AA", "#EE7733", "#228833", "#AA3377",
                  "#66CCEE", "#CCBB44", "#BBBBBB", "#000000")


def plot_cluster_count_comparison(k_comparison):
    """Сравнивает качество, инерцию и ARI без изменения исходных шкал."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    axes[0].errorbar(
        k_comparison.index, k_comparison["silhouette_mean"],
        yerr=k_comparison["silhouette_std"], marker="o", color="#4477AA",
    )
    axes[0].set(title="Silhouette по трём seed", ylabel="Средний силуэт")
    axes[1].plot(k_comparison.index, k_comparison["inertia"], "o-", color="#4477AA")
    axes[1].set(title="Инерция исходного запуска", ylabel="Сумма квадратов расстояний")
    axes[2].plot(k_comparison.index, k_comparison["ari_min"], "o-", color="#4477AA")
    axes[2].set(title="Минимальный ARI между seed", ylabel="ARI")
    for ax in axes:
        ax.set(xlabel="Число кластеров k", xticks=k_comparison.index)
        ax.grid(alpha=0.2)
    return fig, axes



def plot_candidate_cluster_profiles(X, candidate_labels, candidate_ks):
    """Сопоставляет медианные формы при выбранных k."""
    fig, axes = plt.subplots(
        len(candidate_ks), 1, figsize=(12, 3.2 * len(candidate_ks)),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False,
    )
    hours = np.arange(96) / 4
    cluster_colors = CLUSTER_COLORS
    for ax, k in zip(axes.ravel(), candidate_ks):
        # Для сравнения k используем те же медианы нормированных профилей.
        groups = X.groupby(candidate_labels[k]).median()
        sizes = candidate_labels[k].value_counts()
        for cluster, profile in groups.iterrows():
            ax.plot(
                hours, profile, color=cluster_colors[int(cluster)],
                label=f"Кластер {cluster}: {sizes[cluster]} клиентов",
            )
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(title=f"k={k}", ylabel="Нормированная мощность", ylim=(0, None))
        ax.legend(loc="upper right", ncol=2)
    axes[-1, 0].set(xlabel="Время суток", xticks=range(0, 25, 3), xlim=(0, 24))
    return fig, axes



def plot_cluster_profiles(cluster_profiles, cluster_q25, cluster_q75, cluster_sizes):
    """Показывает медиану и межклиентский IQR каждого кластера."""
    hours = np.arange(96) / 4
    cluster_colors = CLUSTER_COLORS
    selected_k = len(cluster_profiles)
    fig, axes = plt.subplots(
        selected_k, 1, figsize=(12, 3 * selected_k),
        sharex=True, constrained_layout=True, squeeze=False,
    )
    for ax, cluster in zip(axes.ravel(), cluster_profiles.index):
        color = cluster_colors[int(cluster)]
        profile = cluster_profiles.loc[cluster]
        # Полоса — разброс клиентов внутри группы, не доверительный интервал.
        ax.fill_between(
            hours, cluster_q25.loc[cluster], cluster_q75.loc[cluster],
            color=color, alpha=0.2, label="25–75% клиентов",
        )
        ax.plot(hours, profile, color=color, linewidth=2, label="Медиана")
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(
            title=f"Кластер {cluster}: {cluster_sizes[cluster]} клиентов; своя шкала Y",
            ylabel="Нормированная мощность", ylim=(0, None),
        )
        legend_location = "upper left" if profile.idxmax() >= "12:00" else "upper right"
        ax.legend(loc=legend_location)
        ax.grid(alpha=0.2)
    axes[-1, 0].set(xlabel="Время суток", xticks=range(0, 25, 3), xlim=(0, 24))
    return fig, axes



def plot_cluster_examples(X, cluster_profiles, cluster_members, representative_clients):
    """Показывает прежнюю диагностическую выборку индивидуальных профилей."""
    hours = np.arange(96) / 4
    cluster_colors = CLUSTER_COLORS
    selected_k = len(cluster_profiles)
    fig, axes = plt.subplots(
        selected_k, 1, figsize=(12, 3 * selected_k),
        sharex=True, constrained_layout=True, squeeze=False,
    )
    for ax, cluster in zip(axes.ravel(), cluster_profiles.index):
        members = cluster_members.loc[cluster_members["cluster"].eq(cluster)]
        # Малые группы показываем целиком; для остальных сохраняем целевую выборку.
        if len(members) <= 12:
            clients_to_show = members.index
        else:
            clients_to_show = representative_clients.index[
                representative_clients["cluster"].eq(cluster)
            ]
        farthest = members["distance_to_median_rmse"].idxmax()
        for client in clients_to_show:
            is_farthest = client == farthest
            ax.plot(
                hours, X.loc[client], alpha=0.75,
                color="#CC3311" if is_farthest else "#777777",
                linestyle="--" if is_farthest else "-",
                label=client,
            )
        ax.plot(
            hours, cluster_profiles.loc[cluster], linewidth=2.5,
            color=cluster_colors[int(cluster)], label="Медиана группы",
        )
        ax.set(
            title=f"Кластер {cluster}: показано {len(clients_to_show)} из {len(members)}; своя шкала Y",
            ylabel="Нормированная мощность", ylim=(0, None),
        )
        legend_location = "upper left" if cluster_profiles.loc[cluster].idxmax() >= "12:00" else "upper right"
        ax.legend(loc=legend_location, ncol=3)
    axes[-1, 0].set(xlabel="Время суток", xticks=range(0, 25, 3), xlim=(0, 24))
    return fig, axes



def plot_cluster_silhouettes(cluster_members):
    """Показывает индивидуальные силуэты с сохранением порядка групп."""
    cluster_colors = CLUSTER_COLORS
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    offset = 0
    # Каждый кластер занимает свою полосу; разрыв отделяет соседние группы.
    for cluster, members in cluster_members.groupby("cluster"):
        scores = np.sort(members["silhouette"].to_numpy())
        y = np.arange(offset, offset + len(scores))
        ax.barh(y, scores, height=1, color=cluster_colors[int(cluster)],
                label=f"Кластер {cluster}, n={len(scores)}")
        offset += len(scores) + 8
    ax.axvline(0, color="gray", linewidth=1)
    ax.axvline(cluster_members["silhouette"].mean(), color="black",
               linestyle="--", label="Средний silhouette")
    ax.set(
        title="Индивидуальные силуэты: все включённые клиенты",
        xlabel="Silhouette", ylabel="Клиенты, отсортированные внутри групп",
        yticks=[], xlim=(-1, 1),
    )
    ax.legend(loc="lower right")
    return fig, ax



def plot_cluster_energy_summary(cluster_summary):
    """Сопоставляет размер группы и её долю энергии всех клиентов."""
    cluster_colors = CLUSTER_COLORS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, column, title, ylabel in (
        (axes[0], "clients", "Число клиентов", "Клиенты"),
        (axes[1], "energy_share_all_pct", "Доля энергии всех клиентов", "% энергии за весь файл"),
    ):
        values = cluster_summary[column]
        bars = ax.bar(
            values.index.astype(str), values,
            color=[cluster_colors[int(c)] for c in values.index],
        )
        ax.bar_label(bars, fmt="%.2f" if column.endswith("pct") else "%.0f")
        ax.set(title=title, xlabel="Кластер", ylabel=ylabel, ylim=(0, values.max() * 1.18))
    return fig, axes
