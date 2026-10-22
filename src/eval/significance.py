"""Значима ли разница между двумя методами? Парный bootstrap по предложениям.

Идея: много раз (n_samples) случайно выбираем предложения тестовой выборки
с возвращением и смотрим, как часто метод B оказывается НЕ лучше метода A.
Эта доля и есть p-value. Если p < 0.05 — улучшение статистически значимо.
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


def paired_bootstrap(sents_a, sents_b, task, n_samples=10_000, seed=42):
    check_same_test_set(sents_a, sents_b)
    stats_a = np.array(sentence_stats(sents_a, task))   # форма: (предложения, 3)
    stats_b = np.array(sentence_stats(sents_b, task))
    n = len(stats_a)

    def score(stats, idx):
        a, b, c = stats[idx].sum(axis=0)
        return score_from_stats(a, b, c, task)

    everything = np.arange(n)
    score_a, score_b = score(stats_a, everything), score(stats_b, everything)

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)   # одни и те же предложения для A и B — «парный»
        diffs.append(score(stats_b, idx) - score(stats_a, idx))
    diffs.sort()

    return {
        "score_a": float(score_a),
        "score_b": float(score_b),
        "diff": float(score_b - score_a),
        "ci95": (float(diffs[int(0.025 * n_samples)]), float(diffs[int(0.975 * n_samples) - 1])),
        "p_value": sum(d <= 0 for d in diffs) / n_samples,   # H0: B не лучше A
        "n_samples": n_samples,
    }
