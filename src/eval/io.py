"""Чтение и запись файлов предсказаний.

ЕДИНЫЙ ФОРМАТ для всех 9 комбинаций (задача × метод) — JSONL,
одна строка = одно предложение:

    {"id": "s001", "tokens": ["Астана", "қаласы"], "gold": ["B-GPE", "O"], "pred": ["B-GPE", "O"]}

- NER:   метки IOB2           ("B-PERSON", "I-PERSON", "O")
- POS:   теги UPOS            ("NOUN", "VERB", ...)
- morph: признаки UD FEATS    ("Case=Nom|Number=Sing", "_" если признаков нет)

Имя файла задаёт, что это за эксперимент:
    {task}_{method}[_{run}].jsonl
    ner_rule.jsonl, pos_neural_seed13.jsonl, morph_hybrid_seed42.jsonl
"""

import json
from pathlib import Path

TASKS = ("ner", "pos", "morph")
METHODS = ("rule", "neural", "hybrid")
REQUIRED_FIELDS = ("tokens", "gold", "pred")


def write_predictions(path, sentences):
    """Сохранить предсказания. sentences — список словарей с tokens, gold, pred (и желательно id)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            record = {
                "id": s.get("id", i),
                "tokens": list(s["tokens"]),
                "gold": list(s["gold"]),
                "pred": list(s["pred"]),
            }
            # ensure_ascii=False — чтобы казахские буквы остались читаемыми
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_predictions(path):
    """Прочитать предсказания и проверить, что файл корректен."""
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in record]
            if missing:
                raise ValueError(f"{path}, строка {line_no}: нет полей {missing}")
            n = len(record["tokens"])
            if not (len(record["gold"]) == n == len(record["pred"])):
                raise ValueError(
                    f"{path}, строка {line_no}: разная длина tokens/gold/pred "
                    f"({n}/{len(record['gold'])}/{len(record['pred'])})"
                )
            sentences.append(record)
    if not sentences:
        raise ValueError(f"{path}: файл пустой")
    return sentences


def parse_name(path):
    """'pos_neural_seed13.jsonl' -> ('pos', 'neural', 'seed13')."""
    parts = Path(path).stem.split("_")
    if len(parts) < 2 or parts[0] not in TASKS or parts[1] not in METHODS:
        raise ValueError(
            f"Имя '{Path(path).name}' не по шаблону {{task}}_{{method}}[_{{run}}].jsonl; "
            f"task из {TASKS}, method из {METHODS}"
        )
    run = "_".join(parts[2:]) or "default"
    return parts[0], parts[1], run
