"""Запуск гибрида: подбор параметров на dev, применение к test, абляция.

    python -m src.hybrid.run --thesis /content/drive/MyDrive/thesis --seed 13

Нужны файлы (в thesis/outputs/preds/{dev,test}/):
    {task}_rule.jsonl                  — python -m src.rules.run
    {task}_neural_seed{N}.jsonl        — python -m src.neural.train
Пишет {task}_hybrid_seed{N}.jsonl и thesis/outputs/hybrid/params_seed{N}.json
(выбранные параметры + абляция: вклад каждого компонента на dev).
"""

import argparse
import json
from itertools import product
from pathlib import Path

from ..eval.io import read_predictions
from ..eval.metrics import MAIN_METRIC, evaluate
from .combine import hybrid_ner, hybrid_sequence_tagging

LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
NER_GRID = {"tau_p": [0.7, 0.85, 0.95, 2.0], "gamma": [0.6, 0.8, 0.95, 1.01], "tau_g": [0.8, 0.9, 0.95, 2.0]}
FIELD = {"pos": "upos", "morph": "feats"}


def _score(rows, task):
    return evaluate(rows, task)[MAIN_METRIC[task]]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tune_and_apply(preds_root, task, seed):
    load = lambda sp, m: read_predictions(preds_root / sp / f"{task}_{m}.jsonl")
    nd, rd = load("dev", f"neural_seed{seed}"), load("dev", "rule")
    nt, rt = load("test", f"neural_seed{seed}"), load("test", "rule")
    report = {"task": task, "seed": seed,
              "dev_neural": _score(nd, task), "dev_rule": _score(rd, task)}

    if task == "ner":
        grid = []
        for tp, g, tg in product(*NER_GRID.values()):
            grid.append(((tp, g, tg), _score(hybrid_ner(nd, rd, tp, g, tg), task)))
        (tp, g, tg), best = max(grid, key=lambda x: x[1])
        report["params"] = {"tau_p": tp, "gamma": g, "tau_g": tg}
        report["dev_hybrid"] = best
        report["ablation_dev"] = {
            "only patterns": _score(hybrid_ner(nd, rd, tp, g, 2.0), task),
            "only gazetteer": _score(hybrid_ner(nd, rd, 2.0, g, tg), task),
            "only IOB2 repair": _score(hybrid_ner(nd, rd, 2.0, g, 2.0), task),
        }
        test_rows = hybrid_ner(nt, rt, tp, g, tg)
        dev_rows = hybrid_ner(nd, rd, tp, g, tg)
    else:
        field = FIELD[task]
        grid = [(lam, _score(hybrid_sequence_tagging(nd, rd, field, lam)[0], task)) for lam in LAMBDAS]
        lam, best = max(grid, key=lambda x: x[1])
        report["params"] = {"lambda": lam}
        report["dev_hybrid"] = best
        report["lambda_curve_dev"] = dict(grid)
        report["ablation_dev"] = {"only hard constraints (λ=0)": grid[0][1]}
        dev_rows, _ = hybrid_sequence_tagging(nd, rd, field, lam)
        test_rows, decisions = hybrid_sequence_tagging(nt, rt, field, lam)
        report["test_decisions"] = dict(decisions)
    _write(preds_root / "dev" / f"{task}_hybrid_seed{seed}.jsonl", dev_rows)
    _write(preds_root / "test" / f"{task}_hybrid_seed{seed}.jsonl", test_rows)
    return report


def run(thesis, seed, tasks=("ner", "pos", "morph"), verbose=True):
    thesis = Path(thesis)
    preds_root = thesis / "outputs" / "preds"
    reports = {}
    for task in tasks:
        if not (preds_root / "dev" / f"{task}_neural_seed{seed}.jsonl").exists():
            if verbose:
                print(f"  {task}: нет предсказаний нейросети для seed {seed} — пропускаю")
            continue
        reports[task] = r = tune_and_apply(preds_root, task, seed)
        if verbose:
            print(f"  {task}: dev rule {r['dev_rule']:.4f} | neural {r['dev_neural']:.4f} | "
                  f"hybrid {r['dev_hybrid']:.4f}  params {r['params']}")
    out = thesis / "outputs" / "hybrid"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"params_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=1)
    return reports


def main():
    p = argparse.ArgumentParser(description="Гибрид: нейросеть → коррекция правилами")
    p.add_argument("--thesis", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tasks", nargs="+", default=["ner", "pos", "morph"])
    a = p.parse_args()
    run(a.thesis, a.seed, a.tasks)


if __name__ == "__main__":
    main()
