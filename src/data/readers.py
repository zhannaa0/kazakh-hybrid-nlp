"""Чтение исходных файлов датасетов в простые Python-структуры.

Каждое предложение — словарь. Общие поля для обоих датасетов:
    id      — уникальный идентификатор предложения
    tokens  — список токенов

KazNERD добавляет:
    ner     — теги IOB2 ("B-PERSON", "I-PERSON", "O")

UD-KTB добавляет:
    doc     — исходный текст (для стратификации сплитов)
    text    — исходное предложение строкой
    lemmas, upos, feats — колонки CoNLL-U

Читатели НИЧЕГО не исправляют — только читают и сообщают о проблемах формата.
Исправления делает clean.py, чтобы было видно «до» и «после».
"""

from pathlib import Path

# --------------------------------------------------------------------- KazNERD


def read_iob2(path, id_prefix=None):
    """Файл вида «токен ТЕГ» по строке, пустая строка = конец предложения."""
    path = Path(path)
    id_prefix = id_prefix or path.stem
    sentences, tokens, tags = [], [], []

    def flush():
        if tokens:
            sentences.append({"id": f"{id_prefix}-{len(sentences) + 1:06d}",
                              "tokens": tokens.copy(), "ner": tags.copy()})
            tokens.clear()
            tags.clear()

    with path.open(encoding="utf-8-sig") as f:          # -sig убирает BOM, если он есть
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                flush()
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path.name}:{line_no}: ожидалось «токен ТЕГ», получено {line!r}")
            tokens.append(parts[0])
            tags.append(parts[1])
    flush()
    return sentences


# ---------------------------------------------------------------------- UD-KTB


def read_conllu(path):
    """CoNLL-U → список предложений из синтаксических слов.

    Пропускаем:
      - строки-диапазоны многословных токенов (12-13  Баласыз) — оцениваем по
        синтаксическим словам (бала + сыз), как принято в UD;
      - пустые узлы (8.1) — в KTB их нет, но формат допускает.
    Сколько пропущено — возвращаем в stats, чтобы описать в главе о данных.
    """
    path = Path(path)
    sentences = []
    stats = {"multiword_tokens": 0, "empty_nodes": 0}
    meta, rows = {}, []

    def flush():
        if not rows:
            return
        sid = meta.get("sent_id", f"{path.stem}-{len(sentences) + 1:06d}")
        sentences.append({
            "id": sid,
            "doc": sid.split(":")[0].replace(".tagged.txt", ""),
            "text": meta.get("text", " ".join(r[1] for r in rows)),
            "tokens": [r[1] for r in rows],
            "lemmas": [r[2] for r in rows],
            "upos": [r[3] for r in rows],
            "feats": [r[5] for r in rows],
        })
        rows.clear()
        meta.clear()

    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                flush()
            elif line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    meta[key.strip()] = value.strip()
            else:
                cols = line.split("\t")
                if len(cols) != 10:
                    raise ValueError(f"{path.name}:{line_no}: в CoNLL-U должно быть 10 колонок, "
                                     f"а их {len(cols)}")
                if "-" in cols[0]:
                    stats["multiword_tokens"] += 1
                elif "." in cols[0]:
                    stats["empty_nodes"] += 1
                else:
                    rows.append(cols)
    flush()
    return sentences, stats
