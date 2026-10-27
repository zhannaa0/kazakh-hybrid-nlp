"""ОДИН скрипт оценки для всех 9 комбинаций (3 задачи × 3 метода).

Использование (из корня проекта):

    # оценить всё в папке -> results/metrics/*.json + results/summary_3x3.csv
    python -m src.eval.evaluate results/preds

    # оценить один файл
    python -m src.eval.evaluate results/preds/ner_rule.jsonl

    # значимо ли hybrid лучше neural? (два файла)
    python -m src.eval.evaluate --compare results/preds/ner_neural.jsonl results/preds/ner_hybrid.jsonl

    # все сравнения сразу: 3 пары методов × 3 задачи, все seed, поправка Холма
    python -m src.eval.evaluate results/preds --significance

Из ноутбука то же самое: from src.eval.evaluate import evaluate_dir, compare_files
"""

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from .io import METHODS, TASKS, parse_name, read_predictions
from .metrics import MAIN_METRIC, evaluate
from .significance import check_same_test_set, holm, paired_bootstrap, paired_bootstrap_runs


def evaluate_file(pred_path, out_dir="results"):
    """Оценить один файл предсказаний и сохранить метрики в results/metrics/<имя>.json."""
    task, method, run = parse_name(pred_path)
    sentences = read_predictions(pred_path)
    result = {"task": task, "method": method, "run": run,
              "file": Path(pred_path).name, **evaluate(sentences, task)}

    metrics_dir = Path(out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{Path(pred_path).stem}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _check_tasks_consistent(files):
    """Внутри одной задачи все методы должны оцениваться на одном и том же тесте."""
    reference = {}
    for path in files:
        task = parse_name(path)[0]
        sents = read_predictions(path)
        if task not in reference:
            reference[task] = (path, sents)
            continue
        try:
            check_same_test_set(reference[task][1], sents)
        except ValueError as e:
            warnings.warn(f"{Path(path).name} и {Path(reference[task][0]).name}: {e}. "
                          "Сравнение этих методов некорректно!")


def evaluate_dir(pred_dir, out_dir="results"):
    """Оценить все *.jsonl в папке и собрать таблицу 3×3 по главной метрике."""
    files = sorted(Path(pred_dir).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"В {pred_dir} нет файлов *.jsonl")
    _check_tasks_consistent(files)

    rows = []
    for path in files:
        r = evaluate_file(path, out_dir)
        rows.append({"task": r["task"], "method": r["method"], "run": r["run"],
                     "metric": MAIN_METRIC[r["task"]], "score": r[MAIN_METRIC[r["task"]]]})
    runs = pd.DataFrame(rows)

    # Среднее ± std по запускам (seed'ам) для каждой ячейки 3×3
    agg = (runs.groupby(["task", "method"])["score"]
               .agg(mean="mean", std="std", n_runs="count").reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    agg["cell"] = agg.apply(
        lambda r: f"{100 * r['mean']:.2f}" + (f" ± {100 * r['std']:.2f}" if r["n_runs"] > 1 else ""),
        axis=1)

    table = (agg.pivot(index="task", columns="method", values="cell")
                .reindex(index=[t for t in TASKS if t in set(agg["task"])],
                         columns=[m for m in METHODS if m in set(agg["method"])])
                .fillna("—"))
    table.insert(0, "metric", [MAIN_METRIC[t] for t in table.index])

    out = Path(out_dir)
    runs.to_csv(out / "summary_runs.csv", index=False)
    agg.to_csv(out / "summary_cells.csv", index=False)
    table.to_csv(out / "summary_3x3.csv")
    return table, runs


def compare_files(path_a, path_b, n_samples=10_000):
    """Парный bootstrap: лучше ли метод B, чем метод A, на той же задаче."""
    task_a, task_b = parse_name(path_a)[0], parse_name(path_b)[0]
    if task_a != task_b:
        raise ValueError(f"Нельзя сравнивать разные задачи: {task_a} и {task_b}")
    return paired_bootstrap(read_predictions(path_a), read_predictions(path_b),
                            task_a, n_samples=n_samples)


COMPARISONS = (("neural", "hybrid"), ("rule", "hybrid"), ("rule", "neural"))   # (A, B): лучше ли B, чем A


def compare_all(pred_dir, out_dir="results", n_samples=10_000):
    """Все сравнения методов внутри каждой задачи (все seed) + поправка Холма на всю таблицу."""
    runs = {}
    for path in sorted(Path(pred_dir).glob("*.jsonl")):
        task, method, _ = parse_name(path)
        runs.setdefault((task, method), []).append(read_predictions(path))
    rows = []
    for task in TASKS:
        for a, b in COMPARISONS:
            if (task, a) in runs and (task, b) in runs:
                r = paired_bootstrap_runs(runs[(task, a)], runs[(task, b)], task, n_samples=n_samples)
                rows.append({"task": task, "metric": MAIN_METRIC[task], "A": a, "B": b,
                             "score_A": r["score_a"], "score_B": r["score_b"], "diff": r["diff"],
                             "ci95_low": r["ci95"][0], "ci95_high": r["ci95"][1],
                             "p": r["p_value"], "runs_A": r["n_runs_a"], "runs_B": r["n_runs_b"]})
    if not rows:
        raise ValueError("Нет пар методов для сравнения: нужны файлы хотя бы двух методов одной задачи")
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    table["significant"] = table["p_holm"] < 0.05
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_dir) / "significance.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description="Оценка предсказаний (3 задачи × 3 метода)")
    parser.add_argument("path", nargs="?", help="файл .jsonl или папка с ними")
    parser.add_argument("--out", default="results", help="куда сохранять метрики")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="сравнить два файла")
    parser.add_argument("--samples", type=int, default=10_000, help="число bootstrap-выборок")
    parser.add_argument("--significance", action="store_true",
                        help="вместе с папкой: все попарные сравнения с поправкой Холма")
    args = parser.parse_args()

    if args.compare:
        a, b = args.compare
        r = compare_files(a, b, args.samples)
        print(f"A = {Path(a).name}: {100 * r['score_a']:.2f}")
        print(f"B = {Path(b).name}: {100 * r['score_b']:.2f}")
        print(f"B − A = {100 * r['diff']:+.2f}  "
              f"(95% CI: {100 * r['ci95'][0]:+.2f} … {100 * r['ci95'][1]:+.2f})")
        verdict = "значимо" if r["p_value"] < 0.05 else "НЕ значимо"
        print(f"p-value = {r['p_value']:.4f} → улучшение {verdict} (α = 0.05)")
    elif args.path and Path(args.path).is_dir():
        table, _ = evaluate_dir(args.path, args.out)
        print(table.to_string())
        if args.significance:
            sig = compare_all(args.path, args.out, args.samples)
            show = sig.copy()
            for c in ("score_A", "score_B", "diff", "ci95_low", "ci95_high"):
                show[c] = (100 * show[c]).round(2)
            print("\n" + show.round({"p": 4, "p_holm": 4}).to_string(index=False))
        print(f"\nСохранено в {args.out}/: summary_3x3.csv, metrics/" + (", significance.csv" if args.significance else ""))
    elif args.path:
        r = evaluate_file(args.path, args.out)
        for k, v in r.items():
            if k != "per_class":
                print(f"{k:>14}: {v:.4f}" if isinstance(v, float) else f"{k:>14}: {v}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
