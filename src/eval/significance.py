"""Методичка, табл. 8, этапы 5 и 7: статистический анализ и оценка значимости.

Парный bootstrap по предложениям. Много раз (n_samples) случайно выбираем
предложения тестовой выборки с возвращением и смотрим, как часто метод B
оказывается НЕ лучше метода A. Эта доля и есть p-value (H0: B не лучше A).

Несколько запусков (seed): статистика каждого предложения усредняется по
запускам метода, и bootstrap идёт по предложениям. Так учитывается и случайность
выборки, и разброс между seed, без «лучшего seed».

Множественные сравнения: при 3 задачах × 3 парах методов вероятность случайно
получить хотя бы одно p < 0.05 заметно больше 5%. Поправка Холма (holm) это
исправляет; в диссертации приводите скорректированные p.
"""

import numpy as np

from .metrics import score_from_stats, sentence_stats


def check_same_test_set(sents_a, sents_b):
    """Два файла можно сравнивать, только если в них одни и те же предложения и gold."""
    if len(sents_a) != len(sents_b):
        raise ValueError(f"Разное число предложений: {len(sents_a)} и {len(sents_b)}")
    for i, (a, b) in enumerate(zip(sents_a, sents_b)):
        if a["tokens"] != b["tokens"] or a["gold"] != b["gold"]:
            raise ValueError(f"Предложение #{i} (id={a.get('id')}) различается в двух файлах")


def _mean_stats(runs, task):
    """runs — список запусков (каждый — список предложений) → средняя статистика (предложения, 3)."""
    for r in runs[1:]:
        check_same_test_set(runs[0], r)
    return np.mean([np.array(sentence_stats(r, task), dtype=float) for r in runs], axis=0)


def paired_bootstrap_runs(runs_a, runs_b, task, n_samples=10_000, seed=42):
    """Сравнить метод A (список запусков) с методом B (список запусков)."""
    check_same_test_set(runs_a[0], runs_b[0])
    stats_a, stats_b = _mean_stats(runs_a, task), _mean_stats(runs_b, task)
    n = len(stats_a)

    def score(stats, idx):
        a, b, c = stats[idx].sum(axis=0)
        return score_from_stats(a, b, c, task)

    everything = np.arange(n)
    score_a, score_b = score(stats_a, everything), score(stats_b, everything)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_samples)
    for k in range(n_samples):
        idx = rng.integers(0, n, size=n)   # одни и те же предложения для A и B — «парный»
        diffs[k] = score(stats_b, idx) - score(stats_a, idx)
    diffs.sort()

    return {
        "score_a": float(score_a),
        "score_b": float(score_b),
        "diff": float(score_b - score_a),
        "ci95": (float(diffs[int(0.025 * n_samples)]), float(diffs[int(0.975 * n_samples) - 1])),
        "p_value": float((diffs <= 0).mean()),
        "n_runs_a": len(runs_a),
        "n_runs_b": len(runs_b),
        "n_samples": n_samples,
    }


def paired_bootstrap(sents_a, sents_b, task, n_samples=10_000, seed=42):
    """Сравнение двух одиночных файлов (частный случай: по одному запуску)."""
    return paired_bootstrap_runs([sents_a], [sents_b], task, n_samples, seed)


def holm(p_values):
    """Поправка Холма–Бонферрони. p_values — список; возвращает скорректированные p в том же порядке."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running_max = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted
