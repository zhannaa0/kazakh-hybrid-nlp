"""Запуск rule-based методов на всех трёх задачах.

    python -m src.rules.run --thesis /content/drive/MyDrive/thesis

Пишет в thesis/outputs/preds/{dev,test}/:
    ner_rule.jsonl, pos_rule.jsonl, morph_rule.jsonl
Кроме обязательных полей (tokens, gold, pred) сохраняется то, что нужно гибриду:
    POS/морфология — "candidates": для каждого токена до 8 разборов [UPOS, FEATS, оценка, источник];
    NER            — "rule_spans": [тип, начало, конец, уверенность, источник].
Словари строятся ТОЛЬКО по train; на dev правила настраивались, test — для итоговой оценки.
"""

import argparse
from pathlib import Path

from ..data.prepare import load_processed
from .morphology import MorphAnalyzer, RuleTagger
from .ner import RuleNER

TOP_K = 8


def run_morph(ud, splits):
    tagger = RuleTagger(MorphAnalyzer(ud["train"]))
    out = {}
    for sp in splits:
        pos, morph = [], []
        for s in ud[sp]:
            cands = tagger.candidates(s["tokens"])
            best = [c[0] for c in cands]
            packed = [[[a.upos, a.feats, round(a.score, 3), a.source] for a in c[:TOP_K]] for c in cands]
            base = {"id": s["id"], "tokens": s["tokens"], "candidates": packed}
            pos.append({**base, "gold": s["upos"], "pred": [a.upos for a in best]})
            morph.append({**base, "gold": s["feats"], "pred": [a.feats for a in best]})
        out[sp] = {"pos": pos, "morph": morph}
    return out


def run_ner(kaz, splits):
    ner = RuleNER(kaz["train"])
    out = {}
    for sp in splits:
        rows = []
        for s in kaz[sp]:
            spans = ner.spans(s["tokens"])
            tags = ["O"] * len(s["tokens"])
            for etype, a, b, _, _ in spans:
                tags[a] = f"B-{etype}"
                for k in range(a + 1, b):
                    tags[k] = f"I-{etype}"
            rows.append({"id": s["id"], "tokens": s["tokens"], "gold": s["ner"], "pred": tags,
                         "rule_spans": [[e, a, b, round(c, 3), src] for e, a, b, c, src in spans]})
        out[sp] = rows
    return out


def write_extended(path, rows):
    """Как eval.io.write_predictions, но с дополнительными полями (кандидаты для гибрида)."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(thesis, splits=("dev", "test"), verbose=True):
    proc = Path(thesis) / "data" / "processed"
    out_root = Path(thesis) / "outputs" / "preds"
    ud = load_processed(proc, "udktb")
    kaz = load_processed(proc, "kaznerd")
    morph = run_morph(ud, splits)
    ner = run_ner(kaz, splits)
    for sp in splits:
        write_extended(out_root / sp / "pos_rule.jsonl", morph[sp]["pos"])
        write_extended(out_root / sp / "morph_rule.jsonl", morph[sp]["morph"])
        write_extended(out_root / sp / "ner_rule.jsonl", ner[sp])
        if verbose:
            print(f"  {sp}: {out_root / sp}")
    return out_root


def main():
    p = argparse.ArgumentParser(description="Rule-based методы: NER, POS, морфология")
    p.add_argument("--thesis", required=True)
    p.add_argument("--splits", nargs="+", default=["dev", "test"])
    a = p.parse_args()
    run(a.thesis, a.splits)


if __name__ == "__main__":
    main()
