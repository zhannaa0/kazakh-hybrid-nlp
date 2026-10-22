"""Сгенерировать ИГРУШЕЧНЫЕ предсказания для всех 9 комбинаций.

Зачем: проверить и понять весь конвейер оценки ДО того, как появятся настоящие модели.
Метод «rule» ошибается чаще, «neural» реже, «hybrid» ещё реже — просто чтобы
таблица 3×3 выглядела правдоподобно. Это НЕ результаты экспериментов.

    python scripts/make_demo_preds.py            # -> demo/preds/*.jsonl
    python -m src.eval.evaluate demo/preds --out demo
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.io import write_predictions  # noqa: E402

GOLD = {
    "ner": [
        (["Қасым-Жомарт", "Тоқаев", "Астанада", "сөйледі", "."],
         ["B-PERSON", "I-PERSON", "B-GPE", "O", "O"]),
        (["Назарбаев", "Университеті", "2010", "жылы", "ашылды", "."],
         ["B-ORGANISATION", "I-ORGANISATION", "B-DATE", "I-DATE", "O", "O"]),
        (["Алматы", "облысында", "қар", "жауды", "."],
         ["B-GPE", "I-GPE", "O", "O", "O"]),
    ],
    "pos": [
        (["Мен", "кітап", "оқыдым", "."], ["PRON", "NOUN", "VERB", "PUNCT"]),
        (["Ол", "үлкен", "үйде", "тұрады", "."], ["PRON", "ADJ", "NOUN", "VERB", "PUNCT"]),
        (["Біз", "ертең", "барамыз", "."], ["PRON", "ADV", "VERB", "PUNCT"]),
    ],
    "morph": [
        (["Мен", "кітап", "оқыдым", "."],
         ["Case=Nom|Number=Sing|Person=1", "Case=Nom|Number=Sing",
          "Mood=Ind|Number=Sing|Person=1|Tense=Past", "_"]),
        (["Ол", "үлкен", "үйде", "тұрады", "."],
         ["Case=Nom|Number=Sing|Person=3", "_", "Case=Loc|Number=Sing",
          "Mood=Ind|Number=Sing|Person=3|Tense=Pres", "_"]),
    ],
}

NOISE = {"rule": 0.30, "neural": 0.15, "hybrid": 0.08}   # доля испорченных токенов
REPEAT = 40                                            # размножим, чтобы было на чём считать


def corrupt(labels, task, rate, rng):
    out = []
    for lab in labels:
        if rng.random() >= rate:
            out.append(lab)
        elif task == "ner":
            out.append("O" if lab != "O" else "B-GPE")
        elif task == "pos":
            out.append(rng.choice(["NOUN", "VERB", "ADJ", "ADV", "PRON"]))
        else:
            out.append("_" if lab != "_" else "Case=Nom|Number=Sing")
    return out


def main(out_dir="demo/preds"):
    for task, examples in GOLD.items():
        for method, rate in NOISE.items():
            seeds = [13, 42, 2026] if method != "rule" else [None]   # rule-based детерминирован
            for seed in seeds:
                rng = random.Random(f"{task}-{method}-{seed}")
                sents = [
                    {"id": f"{task}-{k}-{i}", "tokens": toks, "gold": gold,
                     "pred": corrupt(gold, task, rate, rng)}
                    for k in range(REPEAT) for i, (toks, gold) in enumerate(examples)
                ]
                name = f"{task}_{method}" + (f"_seed{seed}" if seed else "")
                write_predictions(Path(out_dir) / f"{name}.jsonl", sents)
    print(f"Готово: {out_dir}/")


if __name__ == "__main__":
    main()
