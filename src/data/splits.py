"""Формирование разбиений train / dev / test.

KazNERD: официальный сплит (train / valid / test) используется как есть;
         valid переименовывается в dev для единообразия.

UD-KTB:  официальный train — всего 31 предложение, нейросеть на нём не обучить.
         Поэтому train + test объединяются и делятся заново 80 / 10 / 10,
         СТРАТИФИЦИРОВАННО по исходному тексту (doc): каждый из 19 текстов
         (новости, Википедия, сказки, Абай…) делится в той же пропорции,
         поэтому жанровый состав всех трёх частей одинаков.
         Деление детерминировано (seed) и сохраняется в splits.json.
"""

import random
from collections import defaultdict

SPLIT_NAMES = ("train", "dev", "test")


def stratified_split(sentences, ratios=(0.8, 0.1, 0.1), seed=42, key="doc"):
    """Вернуть {"train": [...], "dev": [...], "test": [...]} со списками id."""
    if abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Сумма долей должна быть 1")
    groups = defaultdict(list)
    for s in sentences:
        groups[s[key]].append(s["id"])

    rng = random.Random(seed)
    result = {name: [] for name in SPLIT_NAMES}
    for doc in sorted(groups):                         # sorted — порядок не зависит от словаря
        ids = sorted(groups[doc])
        rng.shuffle(ids)
        n = len(ids)
        n_dev = round(n * ratios[1])
        n_test = round(n * ratios[2])
        # Маленький текст (1–4 предложения) целиком уходит в train,
        # иначе в dev/test попадали бы обрывки по одному предложению
        if n < 5:
            n_dev = n_test = 0
        result["test"] += ids[:n_test]
        result["dev"] += ids[n_test:n_test + n_dev]
        result["train"] += ids[n_test + n_dev:]
    return result


def apply_split(sentences, split_ids):
    """Разложить предложения по сплитам согласно спискам id (порядок как в исходных файлах)."""
    where = {sid: name for name, ids in split_ids.items() for sid in ids}
    out = {name: [] for name in split_ids}
    for s in sentences:
        out[where[s["id"]]].append(s)
    return out
