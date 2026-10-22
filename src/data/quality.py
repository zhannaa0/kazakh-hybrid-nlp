"""Проверки качества, которые ничего не меняют, а только находят проблемы:
дубликаты, утечки между сплитами, противоречивая разметка, некорректный IOB2.
"""

from collections import Counter, defaultdict


def _key(sentence):
    return tuple(sentence["tokens"])


def duplicates_within(sentences):
    """Сколько предложений — повторы уже встречавшихся (в том же сплите)."""
    counts = Counter(_key(s) for s in sentences)
    return sum(n - 1 for n in counts.values() if n > 1)


def leakage(train, other):
    """Доля предложений other, которые дословно есть в train."""
    seen = {_key(s) for s in train}
    n = sum(_key(s) in seen for s in other)
    return {"n": n, "share": n / len(other) if other else 0.0}


def conflicting_labels(splits, label_field):
    """Одинаковый текст — разная разметка. splits: {имя_сплита: предложения}.
    Возвращает список примеров для ручного просмотра."""
    variants = defaultdict(set)
    where = defaultdict(list)
    for name, sentences in splits.items():
        for s in sentences:
            variants[_key(s)].add(tuple(s[label_field]))
            where[_key(s)].append(f"{name}:{s['id']}")
    return [{"tokens": list(k), "n_variants": len(v), "ids": where[k]}
            for k, v in variants.items() if len(v) > 1]


def invalid_iob2(tags):
    """Число тегов I-X без начала сущности."""
    bad, prev = 0, "O"
    for tag in tags:
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            bad += 1
        prev = tag
    return bad
