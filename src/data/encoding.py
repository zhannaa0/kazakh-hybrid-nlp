"""Методичка, табл. 8, этапы 2–3 «Coding and structuring» и «Data transformation».

Категориальные метки → целые числа (так их принимает нейросеть) и обратно.
Словари меток строятся ТОЛЬКО по train и сохраняются в label_maps.json — тогда
у всех методов и всех запусков одинаковая нумерация.

Для морфологии сразу два варианта кодирования (решение «целиком или по
признакам» открыто, см. decision log):
  - feats_whole:   вся строка «Case=Nom|Number=Sing» = один класс;
  - feat_values:   каждая категория (Case, Number, …) — свой классификатор,
                   значение «_» = категории у слова нет.
"""

import json
import math
from collections import Counter
from pathlib import Path

IGNORE_INDEX = -100     # метка, которую функция потерь PyTorch пропускает
NONE = "_"


class LabelMap:
    """Двусторонний словарь «метка ↔ число»."""

    def __init__(self, labels):
        self.labels = list(dict.fromkeys(labels))           # порядок сохраняется, дубли убираются
        self.index = {lab: i for i, lab in enumerate(self.labels)}

    def __len__(self):
        return len(self.labels)

    def encode(self, labels, unknown=None):
        """Метки → числа. Неизвестная метка → unknown (или ошибка, если unknown не задан)."""
        out = []
        for lab in labels:
            if lab in self.index:
                out.append(self.index[lab])
            elif unknown is not None:
                out.append(unknown)
            else:
                raise KeyError(f"Метки {lab!r} нет в словаре (строился по train)")
        return out

    def decode(self, ids):
        return [self.labels[i] for i in ids]


# ------------------------------------------------------------ построение

def ner_labels(train):
    """O, затем B-/I- для каждого типа (в алфавитном порядке) — полная схема IOB2."""
    types = sorted({t[2:] for s in train for t in s["ner"] if t != "O"})
    return ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]


def parse_feats(feats):
    """'Case=Nom|Number=Sing' → {'Case': 'Nom', 'Number': 'Sing'}."""
    if feats in (NONE, "", None):
        return {}
    return dict(p.split("=", 1) for p in feats.split("|"))


def join_feats(d):
    """Обратно в строку UD: категории по алфавиту без учёта регистра, как в стандарте."""
    items = [(k, v) for k, v in d.items() if v != NONE]
    return "|".join(f"{k}={v}" for k, v in sorted(items, key=lambda kv: kv[0].lower())) or NONE


def build_label_maps(kaznerd_train, ud_train):
    feats_counter = Counter(f for s in ud_train for f in s["feats"])
    categories = {}
    for s in ud_train:
        for f in s["feats"]:
            for k, v in parse_feats(f).items():
                categories.setdefault(k, set()).add(v)
    return {
        "ner": ner_labels(kaznerd_train),
        "upos": sorted({u for s in ud_train for u in s["upos"]}),
        "feats_whole": [f for f, _ in feats_counter.most_common()],          # частые — первыми
        "feat_values": {k: [NONE] + sorted(v) for k, v in sorted(categories.items())},
    }


def save_label_maps(maps, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=1)


def load_label_maps(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "ner": LabelMap(raw["ner"]),
        "upos": LabelMap(raw["upos"]),
        "feats_whole": LabelMap(raw["feats_whole"]),
        "feat_values": {k: LabelMap(v) for k, v in raw["feat_values"].items()},
    }


# ------------------------------------------- морфология по признакам

def feats_to_columns(feats, feat_values):
    """'Case=Nom' → {'Case': 'Nom', 'Number': '_', ...} по всем категориям train."""
    d = parse_feats(feats)
    return {k: d.get(k, NONE) for k in feat_values}


def columns_to_feats(cols):
    return join_feats(cols)


# --------------------------------------------- дисбаланс классов

def class_weights(counts, scheme="inverse_sqrt", max_weight=10.0):
    """Веса классов для функции потерь (средний вес = 1).

    inverse      — w ∝ 1 / n          (сильная поправка)
    inverse_sqrt — w ∝ 1 / sqrt(n)    (мягкая; по умолчанию)
    none         — все веса 1
    max_weight ограничивает вес очень редких классов (NON_HUMAN: 6 примеров).
    """
    if scheme == "none":
        return {k: 1.0 for k in counts}
    power = {"inverse": 1.0, "inverse_sqrt": 0.5}[scheme]
    raw = {k: (1.0 / max(n, 1)) ** power for k, n in counts.items()}
    mean = sum(raw.values()) / len(raw)
    return {k: min(v / mean, max_weight) for k, v in raw.items()}


# ------------------------------------------- подслова трансформера

def align_to_subwords(word_ids, word_labels, label_all_subwords=False):
    """Перенести метки слов на подслова.

    word_ids — номер слова для каждого подслова (tokenizer(...).word_ids()),
               None для служебных токенов [CLS]/[SEP]/<s>.
    Метку получает первое подслово каждого слова; остальные — IGNORE_INDEX
    (или ту же метку, если label_all_subwords=True).
    """
    aligned, previous = [], None
    for wid in word_ids:
        if wid is None:
            aligned.append(IGNORE_INDEX)
        elif wid != previous:
            aligned.append(word_labels[wid])
        else:
            aligned.append(word_labels[wid] if label_all_subwords else IGNORE_INDEX)
        previous = wid
    return aligned


def words_from_subwords(word_ids, predicted_ids):
    """Обратная операция: предсказание слова = предсказание его первого подслова."""
    out, previous = {}, None
    for wid, pid in zip(word_ids, predicted_ids):
        if wid is not None and wid != previous:
            out[wid] = pid
        previous = wid
    return [out[i] for i in sorted(out)]


def entropy_bits(counts):
    """Энтропия распределения классов (бит) — одна цифра «насколько равномерно»."""
    total = sum(counts.values())
    return -sum(n / total * math.log2(n / total) for n in counts.values() if n)
