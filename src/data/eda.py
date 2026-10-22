"""Исследовательский анализ данных (EDA): функции возвращают pandas-таблицы.
Графики строит plots.py, ноутбук только вызывает эти функции.

splits — словарь {"train": [...], "dev": [...], "test": [...]} предложений
в едином формате (см. readers.py).
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

from ..eval.metrics import ner_entities
from .quality import conflicting_labels, invalid_iob2

# Части речи, у которых в KTB почти всегда есть признаки (падеж, форма глагола…).
# ADJ не включён: неизменяемые прилагательные в KTB законно размечены без признаков.
CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "PRON", "AUX", "NUM"}
PUNCT_RE = re.compile(r"^\W+$")


# ------------------------------------------------------------------ размеры

def split_sizes(datasets):
    """datasets: {"KazNERD": splits, "UD-KTB": splits, ...} → таблица размеров."""
    rows = []
    for name, splits in datasets.items():
        for sp, S in splits.items():
            rows.append({"dataset": name, "split": sp, "sentences": len(S),
                         "tokens": sum(len(s["tokens"]) for s in S)})
    df = pd.DataFrame(rows)
    df["share_of_sentences"] = df["sentences"] / df.groupby("dataset")["sentences"].transform("sum")
    return df


# ------------------------------------------------------------ распределения

def entities(tags):
    """Сущности (тип, начало, конец) — та же функция, что и в оценке (одно определение на проект)."""
    return sorted(ner_entities(tags), key=lambda e: e[1])


def ner_class_counts(splits):
    """Число СУЩНОСТЕЙ каждого типа по сплитам (не токенов!)."""
    data = {sp: Counter(e[0] for s in S for e in entities(s["ner"])) for sp, S in splits.items()}
    df = pd.DataFrame(data).fillna(0).astype(int)
    return df.sort_values("train", ascending=False)


def token_label_counts(splits, field):
    data = {sp: Counter(t for s in S for t in s[field]) for sp, S in splits.items()}
    df = pd.DataFrame(data).fillna(0).astype(int)
    return df.sort_values(df.columns[0], ascending=False)


def imbalance_summary(counts):
    """counts — Series «класс → число». Насколько несбалансированы классы."""
    counts = counts[counts > 0].sort_values(ascending=False)
    total = counts.sum()
    return pd.Series({
        "classes": len(counts),
        "largest_class": counts.index[0],
        "largest_share": counts.iloc[0] / total,
        "smallest_class": counts.index[-1],
        "smallest_count": int(counts.iloc[-1]),
        "imbalance_ratio": counts.iloc[0] / counts.iloc[-1],         # во сколько раз крупнейший больше мельчайшего
        "top3_share": counts.iloc[:3].sum() / total,
        "classes_lt_1pct": int((counts / total < 0.01).sum()),
    })


def split_consistency(counts):
    """Доли классов в каждом сплите: если сплиты сбалансированы, доли близки."""
    shares = counts / counts.sum()
    shares["max_abs_diff_pp"] = 100 * (shares.max(axis=1) - shares.min(axis=1))
    return shares


# --------------------------------------------------------- длина предложений

def sentence_lengths(datasets):
    rows = [{"dataset": name, "split": sp, "length": len(s["tokens"])}
            for name, splits in datasets.items() for sp, S in splits.items() for s in S]
    return pd.DataFrame(rows)


def length_outliers(lengths, k=1.5):
    """Выбросы по правилу IQR (Тьюки): длина > Q3 + k·IQR или < Q1 − k·IQR."""
    rows = []
    for name, g in lengths.groupby("dataset"):
        q1, q3 = np.percentile(g["length"], [25, 75])
        hi, lo = q3 + k * (q3 - q1), max(1, q1 - k * (q3 - q1))
        rows.append({"dataset": name, "mean": g["length"].mean(), "median": g["length"].median(),
                     "min": g["length"].min(), "max": g["length"].max(),
                     "upper_fence": hi, "n_long_outliers": int((g["length"] > hi).sum()),
                     "n_short_outliers": int((g["length"] < lo).sum()),
                     "share_outliers": float(((g["length"] > hi) | (g["length"] < lo)).mean())})
    return pd.DataFrame(rows).set_index("dataset")


# ---------------------------------------------------------------- редкие слова

def _words(S, lower=True):
    for s in S:
        for t in s["tokens"]:
            if not PUNCT_RE.match(t):
                yield t.lower() if lower else t


def vocabulary_stats(splits):
    """Словарь train, доля hapax (слов, встреченных 1 раз), OOV в dev/test."""
    freq = Counter(_words(splits["train"]))
    rows = [{"split": "train", "types": len(freq), "tokens": sum(freq.values()),
             "hapax_types_share": sum(1 for c in freq.values() if c == 1) / len(freq),
             "rare_le5_types_share": sum(1 for c in freq.values() if c <= 5) / len(freq),
             "oov_token_rate": 0.0, "oov_type_rate": 0.0}]
    for sp in ("dev", "test"):
        if sp not in splits:
            continue
        toks = list(_words(splits[sp]))
        types = set(toks)
        rows.append({"split": sp, "types": len(types), "tokens": len(toks),
                     "hapax_types_share": np.nan, "rare_le5_types_share": np.nan,
                     "oov_token_rate": sum(t not in freq for t in toks) / len(toks),
                     "oov_type_rate": sum(t not in freq for t in types) / len(types)})
    return pd.DataFrame(rows).set_index("split")


def frequency_buckets(splits):
    """Какая доля словоупотреблений приходится на слова с частотой 1, 2–5, 6–20, 21–100, >100."""
    freq = Counter(_words(splits["train"]))
    bins = [(1, 1, "1"), (2, 5, "2–5"), (6, 20, "6–20"), (21, 100, "21–100"), (101, 10**12, ">100")]
    total_types, total_tokens = len(freq), sum(freq.values())
    rows = []
    for lo, hi, label in bins:
        sel = [c for c in freq.values() if lo <= c <= hi]
        rows.append({"frequency": label, "types_share": len(sel) / total_types,
                     "tokens_share": sum(sel) / total_tokens})
    return pd.DataFrame(rows).set_index("frequency")


# -------------------------------------------- пропуски и ошибки в разметке

def ner_annotation_gaps(splits, min_count=5):
    """Возможные пропуски разметки NER: одна и та же строка (≥2 слов или с заглавной)
    размечена как сущность в одних местах и полностью «O» — в других.

    Возвращает таблицу для ручной проверки: строка, тип, сколько раз сущность,
    сколько раз без разметки. Это КАНДИДАТЫ, а не доказанные ошибки:
    например «Астана» может быть и GPE, и частью названия организации.
    """
    # Заглавная буква в начале предложения ничего не говорит об имени собственном
    # («Бір күні…»), поэтому однословные кандидаты в позиции 0 не учитываем.
    def counted(key, start):
        return len(key) > 1 or start > 0

    as_entity = Counter()
    types = {}
    all_sentences = [s for S in splits.values() for s in S]
    for s in all_sentences:
        for etype, a, b in entities(s["ner"]):
            key = tuple(s["tokens"][a:b])
            if counted(key, a):
                as_entity[key] += 1
                types.setdefault(key, Counter())[etype] += 1
    candidates = {k for k, n in as_entity.items()
                  if n >= min_count and (len(k) > 1 or k[0][:1].isupper())}
    by_len = {}
    for k in candidates:
        by_len.setdefault(len(k), set()).add(k)

    as_o = Counter()
    for s in all_sentences:
        toks, tags = s["tokens"], s["ner"]
        for n, keys in by_len.items():
            for i in range(len(toks) - n + 1):
                key = tuple(toks[i:i + n])
                if key in keys and counted(key, i) and all(t == "O" for t in tags[i:i + n]):
                    as_o[key] += 1
    rows = [{"string": " ".join(k), "main_type": types[k].most_common(1)[0][0],
             "as_entity": as_entity[k], "as_O": as_o[k],
             "share_unlabeled": as_o[k] / (as_o[k] + as_entity[k])}
            for k in candidates if as_o[k] > 0]
    if not rows:
        return pd.DataFrame(columns=["string", "main_type", "as_entity", "as_O", "share_unlabeled"])
    return pd.DataFrame(rows).sort_values("as_O", ascending=False).reset_index(drop=True)


def ner_quality_summary(raw_splits, clean_splits):
    """Сводка до/после очистки: некорректные IOB2-теги и противоречия."""
    rows = []
    for stage, splits in (("raw", raw_splits), ("cleaned", clean_splits)):
        rows.append({"stage": stage,
                     "invalid_I_tags": sum(invalid_iob2(s["ner"]) for S in splits.values() for s in S),
                     "conflicting_sentences": len(conflicting_labels(splits, "ner"))})
    return pd.DataFrame(rows).set_index("stage")


def ud_annotation_gaps(splits):
    """Пропуски в разметке UD: пустые леммы/UPOS и знаменательные слова без признаков."""
    rows = []
    for sp, S in splits.items():
        n = sum(len(s["tokens"]) for s in S)
        no_feats = Counter(u for s in S for u, f in zip(s["upos"], s["feats"])
                           if f == "_" and u in CONTENT_UPOS)
        content = Counter(u for s in S for u in s["upos"] if u in CONTENT_UPOS)
        rows.append({"split": sp, "words": n,
                     "missing_lemma": sum(l in ("_", "") for s in S for l in s["lemmas"]),
                     "missing_upos": sum(u in ("_", "") for s in S for u in s["upos"]),
                     "content_words_without_feats": sum(no_feats.values()),
                     "share_content_without_feats": sum(no_feats.values()) / max(1, sum(content.values())),
                     "by_upos": dict(no_feats.most_common())})
    return pd.DataFrame(rows).set_index("split")


# ------------------------------------------------------------- морфология

def feature_counts(splits, split="train"):
    """Как часто встречается каждая морфологическая категория (Case, Number, …)."""
    c = Counter(p.split("=")[0] for s in splits[split] for f in s["feats"] if f != "_"
                for p in f.split("|"))
    return pd.Series(c).sort_values(ascending=False)


def feats_label_space(splits):
    """Сколько разных полных строк FEATS (это число классов для морф-теггера) и сколько из них редки."""
    train = Counter(f for s in splits["train"] for f in s["feats"])
    rows = {"distinct_feats_strings": len(train),
            "singletons": sum(1 for c in train.values() if c == 1),
            "singletons_share": sum(1 for c in train.values() if c == 1) / len(train)}
    for sp in ("dev", "test"):
        if sp in splits:
            toks = [f for s in splits[sp] for f in s["feats"]]
            rows[f"{sp}_unseen_feats_token_rate"] = sum(f not in train for f in toks) / len(toks)
    return pd.Series(rows)


# ------------------------------------------------------- прочие выбросы

def token_outliers(splits, max_len=25):
    """Необычные токены: очень длинные, смешанные алфавиты, с цифрами."""
    rows = []
    for sp, S in splits.items():
        toks = [t for s in S for t in s["tokens"]]
        rows.append({"split": sp, "tokens": len(toks),
                     f"longer_than_{max_len}": sum(len(t) > max_len for t in toks),
                     "with_digits": sum(any(ch.isdigit() for ch in t) for t in toks),
                     "latin_only": sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z\-\.]*", t)) for t in toks),
                     "mixed_scripts": sum(bool(re.search(r"[A-Za-z]", t) and re.search(r"[Ѐ-ӿ]", t))
                                          for t in toks),
                     "punct_only_sentences": sum(all(PUNCT_RE.match(t) for t in s["tokens"]) for s in S)})
    return pd.DataFrame(rows).set_index("split")


def ud_domain_mix(splits, top=5):
    """Состав сплитов по исходным текстам (топ-N, остальное — Other)."""
    total = Counter(s["doc"] for S in splits.values() for s in S)
    keep = [d for d, _ in total.most_common(top)]
    data = {sp: Counter(s["doc"] if s["doc"] in keep else "Other" for s in S) for sp, S in splits.items()}
    df = pd.DataFrame(data).fillna(0).astype(int).reindex(keep + ["Other"])
    return df / df.sum()
