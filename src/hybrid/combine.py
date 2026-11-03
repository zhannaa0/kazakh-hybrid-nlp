"""Гибрид (решение «Option B»): сначала нейросеть, затем коррекция правилами.

POS и морфология — для каждого слова:
  1. Жёсткие ограничения: пунктуация и числа, распознанные правилами
     (источник punct / number), берутся у правил.
  2. Мягкая коррекция: среди кандидатов (нейросеть ∪ правила) выбирается
         ŷ = argmax_y [ log p_θ(y | x) + λ · log q_R(y | x) ]
     p_θ — вероятность нейросети, q_R — распределение правил: softmax оценок
     кандидатов анализатора (ε для разборов, которых правила не предлагали).
     λ = 0 — чистая нейросеть; большое λ — фактически правила. λ подбирается на dev.
     Для FEATS по категориям log p_θ(y) = Σ_k log p_θ(значение категории k).

NER — для каждого предложения:
  1. Шаблонные сущности правил (числа, деньги, проценты, даты) с уверенностью ≥ τ_p
     заменяют разметку нейросети, если нейросеть на этом отрезке не уверена
     (средняя max-вероятность < γ).
  2. Сущности газеттира с надёжностью ≥ τ_g добавляются туда, где нейросеть
     ничего не нашла (повышение полноты).
  3. Исправление IOB2.
τ_p, γ, τ_g подбираются на dev по сетке.
"""

import math

from ..data.clean import repair_iob2
from ..data.encoding import parse_feats

EPS = 1e-4
HARD_SOURCES = {"punct", "number"}


# ======================================================== POS и морфология

def _rule_distribution(cands, field):
    """Кандидаты правил [[upos, feats, score, source], …] → {метка: вероятность}."""
    best = {}
    for upos, feats, score, _ in cands:
        y = upos if field == "upos" else feats
        best[y] = max(best.get(y, -1e9), score)
    if not best:
        return {}
    m = max(best.values())
    z = sum(math.exp(s - m) for s in best.values())
    return {y: math.exp(s - m) / z for y, s in best.items()}


def _neural_logprob(y, probs, mode):
    """log p_θ(y). probs: top-k [[метка, p], …] или, для FEATS по категориям, {кат: [[знач, p], …]}."""
    if mode == "split":
        d = parse_feats(y)
        total = 0.0
        for cat, dist in probs.items():
            v = d.get(cat, "_")
            p = next((pp for val, pp in dist if val == v), EPS)
            total += math.log(max(p, EPS))
        # категории, которых нейросеть не знает (не было в train), — штраф
        total += sum(math.log(EPS) for cat in d if cat not in probs)
        return total
    p = next((pp for lab, pp in probs if lab == y), EPS)
    return math.log(max(p, EPS))


def combine_token(neural_pred, neural_probs, rule_cands, field, lam, mode="top"):
    """Итоговая метка одного слова. field: 'upos' или 'feats'."""
    if rule_cands and rule_cands[0][3] in HARD_SOURCES:
        return rule_cands[0][0] if field == "upos" else rule_cands[0][1], "hard"
    if lam == 0 or not neural_probs:
        return neural_pred, "neural"
    q = _rule_distribution(rule_cands, field)
    options = set(q) | {neural_pred}
    if mode != "split":
        options |= {lab for lab, _ in neural_probs}
    scored = {y: _neural_logprob(y, neural_probs, mode) + lam * math.log(q.get(y, EPS)) for y in options}
    best = max(scored, key=scored.get)
    return best, ("neural" if best == neural_pred else "rule_corrected")


def hybrid_sequence_tagging(neural_rows, rule_rows, field, lam):
    """Применить гибрид ко всему файлу. Возвращает строки предсказаний и счётчик решений."""
    from collections import Counter
    out, decisions = [], Counter()
    for n, r in zip(neural_rows, rule_rows):
        if n["tokens"] != r["tokens"]:
            raise ValueError(f"Разные предложения в файлах нейросети и правил: {n.get('id')}")
        mode = n.get("morph_mode", "top") if field == "feats" else "top"
        pred = []
        for i in range(len(n["tokens"])):
            y, how = combine_token(n["pred"][i], n["probs"][i], r["candidates"][i], field, lam, mode)
            pred.append(y)
            decisions[how] += 1
        out.append({"id": n["id"], "tokens": n["tokens"], "gold": n["gold"], "pred": pred})
    return out, decisions


# ================================================================ NER

def _spans(tags):
    from ..eval.metrics import ner_entities
    return ner_entities(tags)


def hybrid_ner_sentence(neural_tags, neural_probs, rule_spans, tau_p, gamma, tau_g):
    tags = list(neural_tags)
    conf = [max((p for _, p in pr), default=0.0) for pr in neural_probs]
    for etype, a, b, c, source in rule_spans:
        span_conf = sum(conf[a:b]) / (b - a)
        if source == "pattern" and c >= tau_p and span_conf < gamma:
            # отрезок правила заменяет разметку нейросети; соседние «обрубки» починит repair_iob2
            tags[a:b] = [f"B-{etype}"] + [f"I-{etype}"] * (b - a - 1)
            if b < len(tags) and tags[b].startswith("I-"):
                tags[b] = "B-" + tags[b][2:]
        elif source == "gazetteer" and c >= tau_g and all(t == "O" for t in tags[a:b]):
            tags[a:b] = [f"B-{etype}"] + [f"I-{etype}"] * (b - a - 1)
    return repair_iob2(tags)


def hybrid_ner(neural_rows, rule_rows, tau_p, gamma, tau_g):
    out = []
    for n, r in zip(neural_rows, rule_rows):
        if n["tokens"] != r["tokens"]:
            raise ValueError(f"Разные предложения в файлах нейросети и правил: {n.get('id')}")
        pred = hybrid_ner_sentence(n["pred"], n["probs"], r["rule_spans"], tau_p, gamma, tau_g)
        out.append({"id": n["id"], "tokens": n["tokens"], "gold": n["gold"], "pred": pred})
    return out
