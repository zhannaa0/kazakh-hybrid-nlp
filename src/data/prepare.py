"""Полный конвейер подготовки данных: чтение → очистка → сплиты → сохранение.

Запуск (из корня проекта):
    python -m src.data.prepare --raw /content/drive/MyDrive/thesis/data \
                               --out /content/drive/MyDrive/thesis/data/processed

Ожидаемая структура --raw (как у вас на Drive):
    kaznerd/IOB2_train.txt, IOB2_valid.txt, IOB2_test.txt
    ud-ktb/kk_ktb-ud-train.conllu, kk_ktb-ud-test.conllu

Результат в --out:
    kaznerd/{train,dev,test}.jsonl        — единый формат, после очистки
    udktb/{train,dev,test}.jsonl          — новый сплит 80/10/10
    udktb/official/{train,test}.jsonl     — официальный сплит (для справки)
    splits.json                           — какие id в каком сплите UD-KTB
    label_maps.json                       — нумерация меток (по train) для всех методов
    cleaning_log.jsonl                    — каждое исправление: было → стало
    manifest.json                         — ФИКСАЦИЯ ВЕРСИЙ: хэши исходных и
                                            выходных файлов, версии, параметры, счётчики
"""

import argparse
import copy
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .clean import clean_kaznerd, clean_ud
from .collect import SOURCES, sha256
from .encoding import build_label_maps, save_label_maps
from .quality import conflicting_labels, duplicates_within, invalid_iob2, leakage
from .readers import read_conllu, read_iob2
from .splits import apply_split, stratified_split

KAZNERD_FILES = {"train": "IOB2_train.txt", "dev": "IOB2_valid.txt", "test": "IOB2_test.txt"}
UD_FILES = {"train": "kk_ktb-ud-train.conllu", "test": "kk_ktb-ud-test.conllu"}

PREPROCESSING_VERSION = "1.0"   # повышайте, если меняются правила очистки или сплиты


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_processed(out_dir, dataset, splits=("train", "dev", "test")):
    """Удобная загрузка подготовленных данных: load_processed(OUT, 'kaznerd')."""
    return {sp: read_jsonl(Path(out_dir) / dataset / f"{sp}.jsonl") for sp in splits}


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return None


def _counts(splits, label_field):
    return {sp: {"sentences": len(S), "tokens": sum(len(s["tokens"]) for s in S),
                 "labels": len({t for s in S for t in s[label_field]})}
            for sp, S in splits.items()}


def load_raw(raw_dir):
    """Прочитать исходные файлы без изменений (нужно и для EDA «до»)."""
    raw_dir = Path(raw_dir)
    kaz = {sp: read_iob2(raw_dir / "kaznerd" / fn, id_prefix=f"kaznerd-{sp}")
           for sp, fn in KAZNERD_FILES.items()}
    ud, ud_stats = {}, Counter()
    for sp, fn in UD_FILES.items():
        ud[sp], stats = read_conllu(raw_dir / "ud-ktb" / fn)
        ud_stats.update(stats)
    return kaz, ud, dict(ud_stats)


def prepare(raw_dir, out_dir, seed=42, ratios=(0.8, 0.1, 0.1)):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    kaz_raw, ud_raw, ud_format_stats = load_raw(raw_dir)
    log = []

    # ---- KazNERD: чистим, сплиты официальные
    kaz = copy.deepcopy(kaz_raw)
    invalid_before = {sp: sum(invalid_iob2(s["ner"]) for s in S) for sp, S in kaz.items()}
    for sp, S in kaz.items():
        clean_kaznerd(S, sp, log)
    for sp, S in kaz.items():
        write_jsonl(out_dir / "kaznerd" / f"{sp}.jsonl", S)

    # ---- UD-KTB: чистим, сохраняем официальный сплит, делаем новый
    ud = copy.deepcopy(ud_raw)
    for sp, S in ud.items():
        clean_ud(S, f"official-{sp}", log)
        write_jsonl(out_dir / "udktb" / "official" / f"{sp}.jsonl", S)
    pool = ud["train"] + ud["test"]
    split_ids = stratified_split(pool, ratios=ratios, seed=seed)
    ud_new = apply_split(pool, split_ids)
    for sp, S in ud_new.items():
        write_jsonl(out_dir / "udktb" / f"{sp}.jsonl", S)

    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"kaznerd": "official (valid → dev)",
                   "udktb": {"method": "stratified by source document", "seed": seed,
                             "ratios": ratios, "ids": split_ids}}, f, ensure_ascii=False, indent=1)
    write_jsonl(out_dir / "cleaning_log.jsonl", log)

    # ---- Кодирование меток (табл. 8, этап 2): словари только по train
    label_maps = build_label_maps(kaz["train"], ud_new["train"])
    save_label_maps(label_maps, out_dir / "label_maps.json")

    # ---- Проверки качества (ничего не меняют)
    quality = {
        "kaznerd": {
            "invalid_iob2_before": invalid_before,
            "duplicates_within": {sp: duplicates_within(S) for sp, S in kaz.items()},
            "leakage_from_train": {sp: leakage(kaz["train"], kaz[sp]) for sp in ("dev", "test")},
            "conflicting_labels": len(conflicting_labels(kaz, "ner")),
        },
        "udktb": {
            "format": ud_format_stats,
            "duplicates_within": {sp: duplicates_within(S) for sp, S in ud_new.items()},
            "leakage_from_train": {sp: leakage(ud_new["train"], ud_new[sp]) for sp in ("dev", "test")},
            "conflicting_upos": len(conflicting_labels(ud_new, "upos")),
        },
    }

    # ---- Манифест = фиксация версии данных
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "code_commit": _git_commit(),
        "sources": {
            key: {k: src[k] for k in ("name", "repo", "version", "license", "citation")} |
                 {"files": {fn: sha256(raw_dir / key / fn) for fn in src["files"]},
                  "matches_pinned_version": all(sha256(raw_dir / key / fn) == h
                                                for fn, h in src["files"].items())}
            for key, src in SOURCES.items()
        },
        "params": {"seed": seed, "ud_ratios": ratios},
        "counts": {"kaznerd": _counts(kaz, "ner"),
                   "udktb": _counts(ud_new, "upos"),
                   "udktb_official": _counts(ud, "upos")},
        "cleaning": dict(Counter(r["rule"] for r in log)),
        "label_space": {"ner_tags": len(label_maps["ner"]), "upos": len(label_maps["upos"]),
                        "feats_whole": len(label_maps["feats_whole"]),
                        "feat_categories": len(label_maps["feat_values"])},
        "quality": quality,
        "outputs": {str(p.relative_to(out_dir)): sha256(p)
                    for p in sorted(out_dir.rglob("*.jsonl")) if p.name != "cleaning_log.jsonl"},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main():
    p = argparse.ArgumentParser(description="Подготовка KazNERD и UD-KTB")
    p.add_argument("--raw", required=True, help="папка с kaznerd/ и ud-ktb/")
    p.add_argument("--out", required=True, help="куда сохранить подготовленные данные")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    m = prepare(a.raw, a.out, seed=a.seed)
    print("Готово:", a.out)
    for ds in ("kaznerd", "udktb", "udktb_official"):
        print(f"  {ds:15}", {sp: c["sentences"] for sp, c in m["counts"][ds].items()})
    print("  исправлений:", m["cleaning"])


if __name__ == "__main__":
    main()
