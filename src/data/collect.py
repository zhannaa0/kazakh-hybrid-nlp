"""Методичка, табл. 7, шаг 5 «Data collection»: сбор данных по плану, под контролем качества.

Данные скачиваются из официальных репозиториев по ЗАФИКСИРОВАННЫМ версиям
(коммит / тег), и каждый файл сверяется с эталонным SHA-256. Если файл на Drive
отличается от того, на котором получены результаты диссертации, вы узнаете об этом
сразу, а не на защите.

    python -m src.data.collect --raw /content/drive/MyDrive/thesis/data            # скачать недостающее + проверить
    python -m src.data.collect --raw /content/drive/MyDrive/thesis/data --verify    # только проверить
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

# Источник истины о версиях данных. Меняется только осознанно (и записывается в decision log).
SOURCES = {
    "kaznerd": {
        "name": "KazNERD",
        "repo": "https://github.com/IS2AI/KazNERD",
        "version": "commit bd4333d (2025-01-15)",
        "license": "CC BY 4.0",
        "citation": "Yeshpanov et al., 2022. KazNERD: Kazakh Named Entity Recognition Dataset. LREC.",
        "base_url": "https://raw.githubusercontent.com/IS2AI/KazNERD/"
                    "bd4333d0f5952b9fafb2ef2ac2fefa0ad3c0333f/KazNERD/",
        "files": {
            "IOB2_train.txt": "e5a80f6d4bbff499a4bf7cbf96a169feb761411151fb45abebeaae06995ff647",
            "IOB2_valid.txt": "29d584d1a0d9aca6d58b39eb8a93e154791d303ecff56ef93f12b3c889239bf1",
            "IOB2_test.txt": "359a1f61f6a21280c6c47b4fde9a5adcd07411d62015aa6e32d940c51f163b7a",
        },
    },
    "ud-ktb": {
        "name": "UD Kazakh-KTB",
        "repo": "https://github.com/UniversalDependencies/UD_Kazakh-KTB",
        "version": "release r2.18",
        "license": "CC BY-SA 4.0",
        "citation": "Makazhanov et al., 2015; Tyers & Washington, 2015. UD Kazakh KTB.",
        "base_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Kazakh-KTB/r2.18/",
        "files": {
            "kk_ktb-ud-train.conllu": "6297597914a3f319db573573f927a6577c702f5fa00dcd653162aab46026aeaa",
            "kk_ktb-ud-test.conllu": "1100846ee50c2f970562a21dca8415a381c470ea06bdb8496ae213d59c3a3c4e",
        },
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(raw_dir):
    """Проверить все файлы. Возвращает список строк-отчётов: (датасет, файл, статус)."""
    rows = []
    for key, src in SOURCES.items():
        for fn, expected in src["files"].items():
            p = Path(raw_dir) / key / fn
            if not p.exists():
                status = "missing"
            elif sha256(p) == expected:
                status = "ok"
            else:
                status = "HASH MISMATCH"
            rows.append({"dataset": src["name"], "file": fn, "status": status})
    return rows


def collect(raw_dir, download=True):
    """Скачать недостающие файлы и проверить все. Бросает ошибку, если что-то не сходится."""
    raw_dir = Path(raw_dir)
    for key, src in SOURCES.items():
        (raw_dir / key).mkdir(parents=True, exist_ok=True)
        for fn in src["files"]:
            p = raw_dir / key / fn
            if download and not p.exists():
                print(f"скачиваю {src['name']}: {fn}")
                urllib.request.urlretrieve(src["base_url"] + fn, p)
    report = verify(raw_dir)
    bad = [r for r in report if r["status"] != "ok"]
    if bad:
        raise RuntimeError("Данные не совпадают с зафиксированной версией:\n" +
                           "\n".join(f"  {r['dataset']} / {r['file']}: {r['status']}" for r in bad))
    return report


def main():
    p = argparse.ArgumentParser(description="Сбор и проверка исходных данных")
    p.add_argument("--raw", required=True, help="папка thesis/data")
    p.add_argument("--verify", action="store_true", help="только проверить, не скачивать")
    a = p.parse_args()
    for r in collect(a.raw, download=not a.verify):
        print(f"  {r['status']:>4}  {r['dataset']} / {r['file']}")
    print("Все файлы совпадают с зафиксированными версиями.")


if __name__ == "__main__":
    main()
