"""Метрики для трёх задач. Одни и те же функции для rule-based, neural и hybrid.

Главная метрика задачи (по ней строится таблица 3×3):
    ner   -> f1           (entity-level, strict, IOB2)
    pos   -> accuracy     (по токенам)
    morph -> exact_match  (весь набор признаков токена совпал)
"""

from collections import Counter, defaultdict

from seqeval.metrics import classification_report as seq_report
from seqeval.scheme import IOB2, Entities
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
    Некорректная последовательность (I- без B-) сущностью не считается."""
    # to_tuple() -> (номер_предложения, тип, начало, конец); номер нам не нужен
    return {e.to_tuple()[1:] for e in Entities([list(tags)], IOB2).entities[0]}


def ner_metrics(sentences):
    y_true = [s["gold"] for s in sentences]
    y_pred = [s["pred"] for s in sentences]
    report = seq_report(y_true, y_pred, mode="strict", scheme=IOB2,
                        output_dict=True, zero_division=0)
    per_class = {
        label: {"precision": v["precision"], "recall": v["recall"],
                "f1": v["f1-score"], "support": int(v["support"])}
        for label, v in report.items()
        if label not in ("micro avg", "macro avg", "weighted avg")
    }
    return {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
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
