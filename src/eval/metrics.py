"""Метрики для трёх задач. Одни и те же функции для rule-based, neural и hybrid.

Главная метрика задачи (по ней строится таблица 3×3):
    ner   -> f1           (entity-level, strict, IOB2)
    pos   -> accuracy     (по токенам)
    morph -> exact_match  (весь набор признаков токена совпал)
"""

from collections import Counter, defaultdict

from sklearn.metrics import classification_report as sk_report

MAIN_METRIC = {"ner": "f1", "pos": "accuracy", "morph": "exact_match"}


def _flat(sentences, key):
    return [label for s in sentences for label in s[key]]


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------- NER

def ner_entities(tags):
    """Множество сущностей (тип, начало, конец) в strict-режиме IOB2.

    Сущность = B-X и следующие за ним I-X того же типа; конец не включается.
    Некорректная последовательность (I- без B-, или I-Y после B-X) сущностью
    не считается — так же, как seqeval в mode="strict", scheme=IOB2.
    """
    entities = set()
    start, etype = None, None
    for i, tag in enumerate(list(tags) + ["O"]):      # "O" в конце закрывает последнюю сущность
        prefix, _, label = tag.partition("-")
        if start is not None and not (prefix == "I" and label == etype):
            entities.add((etype, start, i))
            start = None
        if prefix == "B":
            start, etype = i, label
    return entities


def ner_metrics(sentences):
    tp, fp, fn = Counter(), Counter(), Counter()
    for s in sentences:
        G, P = ner_entities(s["gold"]), ner_entities(s["pred"])
        for e in G & P:
            tp[e[0]] += 1
        for e in P - G:
            fp[e[0]] += 1
        for e in G - P:
            fn[e[0]] += 1

    per_class = {}
    for label in sorted(set(tp) | set(fp) | set(fn)):
        p, r, f = _prf(tp[label], fp[label], fn[label])
        per_class[label] = {"precision": p, "recall": r, "f1": f,
                            "support": tp[label] + fn[label]}

    precision, recall, f1 = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro_f1 = (sum(v["f1"] for v in per_class.values()) / len(per_class)) if per_class else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,                 # micro-F1 по сущностям — главная метрика NER
        "macro_f1": macro_f1,
        "n_sentences": len(sentences),
        "per_class": per_class,
    }


# ---------------------------------------------------------------- POS

def pos_metrics(sentences):
    gold, pred = _flat(sentences, "gold"), _flat(sentences, "pred")
    labels = sorted(set(gold))
    report = sk_report(gold, pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        label: {"precision": report[label]["precision"], "recall": report[label]["recall"],
                "f1": report[label]["f1-score"], "support": int(report[label]["support"])}
        for label in labels
    }
    return {
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "macro_f1": report["macro avg"]["f1-score"],
        "n_sentences": len(sentences),
        "n_tokens": len(gold),
        "per_class": per_class,
    }


# ---------------------------------------------------------------- MORPH

def parse_feats(feats):
    """'Case=Nom|Number=Sing' -> {'Case=Nom', 'Number=Sing'}; '_' -> пустое множество."""
    return set() if feats in ("_", "", None) else set(feats.split("|"))


def morph_metrics(sentences):
    gold, pred = _flat(sentences, "gold"), _flat(sentences, "pred")
    exact = sum(parse_feats(g) == parse_feats(p) for g, p in zip(gold, pred)) / len(gold)

    # Пары Признак=Значение: частично верный анализ получает частичный балл
    total = Counter()
    by_feature = defaultdict(Counter)
    for g, p in zip(gold, pred):
        G, P = parse_feats(g), parse_feats(p)
        for kind, items in (("tp", G & P), ("fp", P - G), ("fn", G - P)):
            total[kind] += len(items)
            for item in items:
                by_feature[item.split("=")[0]][kind] += 1

    precision, recall, f1 = _prf(total["tp"], total["fp"], total["fn"])
    per_feature = {}
    for name, c in sorted(by_feature.items()):
        fp_, rc_, f_ = _prf(c["tp"], c["fp"], c["fn"])
        per_feature[name] = {"precision": fp_, "recall": rc_, "f1": f_,
                             "support": c["tp"] + c["fn"]}
    return {
        "exact_match": exact,
        "feat_precision": precision,
        "feat_recall": recall,
        "feat_f1": f1,
        "n_sentences": len(sentences),
        "n_tokens": len(gold),
        "per_class": per_feature,
    }


METRICS = {"ner": ner_metrics, "pos": pos_metrics, "morph": morph_metrics}


def evaluate(sentences, task):
    if task not in METRICS:
        raise ValueError(f"Неизвестная задача: {task}")
    return METRICS[task](sentences)


# ------------------------------------------- статистика по предложениям
# Нужна для bootstrap: главную метрику можно пересчитать из сумм этих чисел,
# не вызывая тяжёлые функции тысячи раз.

def sentence_stats(sentences, task):
    """Для каждого предложения: (a, b, c), из сумм которых считается главная метрика."""
    stats = []
    for s in sentences:
        if task == "ner":
            G, P = ner_entities(s["gold"]), ner_entities(s["pred"])
            stats.append((len(G & P), len(P - G), len(G - P)))     # tp, fp, fn
        elif task == "pos":
            correct = sum(g == p for g, p in zip(s["gold"], s["pred"]))
            stats.append((correct, len(s["gold"]), 0))
        else:
            correct = sum(parse_feats(g) == parse_feats(p) for g, p in zip(s["gold"], s["pred"]))
            stats.append((correct, len(s["gold"]), 0))
    return stats


def score_from_stats(a, b, c, task):
    if task == "ner":
        return _prf(a, b, c)[2]
    return a / b if b else 0.0
