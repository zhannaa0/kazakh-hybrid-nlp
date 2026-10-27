"""Методичка, табл. 8, этап 4 «Aggregation»: результаты по группам.

Общая цифра (F1 = 0.85) не говорит, ГДЕ метод ошибается. Здесь та же метрика
считается отдельно по группам:

  by_oov        — слова/сущности, встречавшиеся в train, против новых (OOV).
                  Главная проверка гипотезы: помогают ли правила на незнакомых словах.
  by_length     — короткие / средние / длинные предложения.
  by_group      — любая группировка предложений, например по исходному тексту UD-KTB.

Работает с теми же файлами предсказаний, что и evaluate.py.
"""

from collections import defaultdict

import pandas as pd

from .metrics import _prf, ner_entities, parse_feats, score_from_stats, sentence_stats


def train_vocabulary(train_sentences):
    return {t.lower() for s in train_sentences for t in s["tokens"]}


def _token_correct(task, gold, pred):
    if task == "morph":
        return parse_feats(gold) == parse_feats(pred)
    return gold == pred


def by_oov(sentences, task, vocab):
    """Метрика отдельно для известных (in-vocabulary) и новых (OOV) слов/сущностей."""
    rows = []
    if task == "ner":
        counts = {g: {"tp": 0, "fp": 0, "fn": 0} for g in ("in_vocab", "oov")}
        for s in sentences:
            low = [t.lower() for t in s["tokens"]]

            def group(e):
                _, a, b = e
                return "in_vocab" if all(t in vocab for t in low[a:b]) else "oov"

            G, P = ner_entities(s["gold"]), ner_entities(s["pred"])
            for e in G & P:
                counts[group(e)]["tp"] += 1
            for e in P - G:
                counts[group(e)]["fp"] += 1
            for e in G - P:
                counts[group(e)]["fn"] += 1
        for g, c in counts.items():
            p, r, f = _prf(c["tp"], c["fp"], c["fn"])
            rows.append({"group": g, "n": c["tp"] + c["fn"], "precision": p, "recall": r, "f1": f})
    else:
        acc = defaultdict(lambda: [0, 0])
        for s in sentences:
            for tok, g, p in zip(s["tokens"], s["gold"], s["pred"]):
                key = "in_vocab" if tok.lower() in vocab else "oov"
                acc[key][0] += _token_correct(task, g, p)
                acc[key][1] += 1
        for g in ("in_vocab", "oov"):
            correct, total = acc[g]
            rows.append({"group": g, "n": total, "score": correct / total if total else float("nan")})
    return pd.DataFrame(rows).set_index("group")


def by_group(sentences, task, key):
    """key(предложение) → имя группы. Главная метрика задачи по каждой группе."""
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for s, st in zip(sentences, sentence_stats(sentences, task)):
        g = sums[key(s)]
        g[0] += st[0]
        g[1] += st[1]
        g[2] += st[2]
        g[3] += 1
    rows = [{"group": k, "sentences": v[3], "score": score_from_stats(v[0], v[1], v[2], task)}
            for k, v in sums.items()]
    return pd.DataFrame(rows).set_index("group").sort_index()


def length_bin(s, edges=(10, 20)):
    n = len(s["tokens"])
    if n <= edges[0]:
        return f"1–{edges[0]}"
    if n <= edges[1]:
        return f"{edges[0] + 1}–{edges[1]}"
    return f">{edges[1]}"


def by_length(sentences, task, edges=(10, 20)):
    return by_group(sentences, task, lambda s: length_bin(s, edges))


def ud_document(s):
    """id предложения UD-KTB «sholpan.tagged.txt:1:0» → «sholpan»."""
    return str(s.get("id", "")).split(":")[0].replace(".tagged.txt", "")
