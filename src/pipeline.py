"""Этап 1 диссертации — данные. Шаги идут в порядке методички AITU
(табл. 7 «Data collection», табл. 8 «Data processing»).

    python -m src.pipeline --thesis /content/drive/MyDrive/thesis

| Шаг методички                     | Функция          | Результат (в thesis/outputs/stage1_data/)        |
|-----------------------------------|------------------|--------------------------------------------------|
| 7.5  Data collection              | step_collect     | исходники проверены по SHA-256                    |
| 8.1  Data quality check           | step_quality     | quality/*.csv — проблемы исходных данных          |
| 8.2  Coding and structuring       | step_prepare     | processed/*.jsonl, label_maps.json                |
| 8.3  Data transformation          | step_prepare     | cleaning_log.jsonl                                |
| 8.4  Aggregation and summation    | step_analyse     | tables/*.csv                                      |
| 8.5  Statistical analysis         | step_analyse     | tables/descriptive_statistics.csv                 |
| 8.6  Data visualisation           | step_visualise   | figures/*.png                                     |
| 8.8  Creating a report            | step_report      | report.md, report.html                            |
| 8.9  Secrecy and confidentiality  | step_document    | лицензии и условия хранения в pipeline_log.json   |
| 8.10 Documentation                | step_document    | pipeline_log.json + processed/manifest.json       |

Шаги 7.1–7.4 (тип данных, план, выбор метода, инструменты) — решения, они описаны
в docs/data.md; 7.7 / 8.7 (анализ и интерпретация результатов) — в src/eval после
появления предсказаний моделей; 8.11 — публикация.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import pandas as pd

from .data import eda, plots
from .data.collect import SOURCES, collect
from .data.encoding import class_weights, entropy_bits
from .data.prepare import load_processed, load_raw, prepare
from .data.quality import conflicting_labels, duplicates_within, invalid_iob2, leakage
from .report import Report


class Context:
    """Пути и промежуточные результаты, которые шаги передают друг другу."""

    def __init__(self, thesis_dir):
        self.thesis = Path(thesis_dir)
        self.raw = self.thesis / "data"
        self.processed = self.thesis / "data" / "processed"
        self.out = self.thesis / "outputs" / "stage1_data"
        self.figures = self.out / "figures"
        self.tables = self.out / "tables"
        self.quality_dir = self.out / "quality"
        for d in (self.figures, self.tables, self.quality_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.r = {}          # результаты шагов

    def save_table(self, name, df, index=True):
        df.to_csv(self.tables / f"{name}.csv", index=index)
        self.r.setdefault("tables", {})[name] = df
        return df


# ------------------------------------------------------------- 7.5

def step_collect(ctx, download=False):
    """Табл. 7, шаг 5: данные по плану и под контролем качества (версии + хэши)."""
    report = collect(ctx.raw, download=download)
    ctx.r["collect"] = pd.DataFrame(report)
    return ctx.r["collect"]


# ------------------------------------------------------------- 8.1

def step_quality(ctx):
    """Табл. 8, этап 1: ошибки, пропуски, несогласованности — в ИСХОДНЫХ данных."""
    kaz, ud, ud_format = load_raw(ctx.raw)
    ctx.r["raw"] = {"kaznerd": kaz, "udktb": ud}

    summary = pd.DataFrame({
        "KazNERD": {
            "invalid IOB2 tags": sum(invalid_iob2(s["ner"]) for S in kaz.values() for s in S),
            "duplicate sentences in train": duplicates_within(kaz["train"]),
            "dev sentences also in train": leakage(kaz["train"], kaz["dev"])["n"],
            "test sentences also in train": leakage(kaz["train"], kaz["test"])["n"],
            "sentences with conflicting labels": len(conflicting_labels(kaz, "ner")),
            "empty tokens or labels": sum(not t or not g for S in kaz.values() for s in S
                                          for t, g in zip(s["tokens"], s["ner"])),
        },
        "UD-KTB": {
            "invalid IOB2 tags": 0,
            "duplicate sentences in train": duplicates_within(ud["train"] + ud["test"]),
            "dev sentences also in train": 0,
            "test sentences also in train": leakage(ud["train"], ud["test"])["n"],
            "sentences with conflicting labels": len(conflicting_labels(ud, "upos")),
            "empty tokens or labels": sum(v in ("", "_") for S in ud.values() for s in S
                                          for v in s["lemmas"] + s["upos"]),
        },
    })
    summary.index.name = "check"
    summary.to_csv(ctx.quality_dir / "summary.csv")

    conflicts = pd.DataFrame([{"tokens": " ".join(c["tokens"]), "variants": c["n_variants"],
                               "ids": "; ".join(c["ids"])} for c in conflicting_labels(kaz, "ner")])
    conflicts.to_csv(ctx.quality_dir / "kaznerd_conflicting_labels.csv", index=False)
    gaps = eda.ner_annotation_gaps(kaz)
    gaps.to_csv(ctx.quality_dir / "kaznerd_annotation_gap_candidates.csv", index=False)
    seen = {tuple(s["tokens"]) for s in kaz["train"]}
    leaked = pd.DataFrame([{"split": sp, "id": s["id"], "text": " ".join(s["tokens"])}
                           for sp in ("dev", "test") for s in kaz[sp] if tuple(s["tokens"]) in seen])
    leaked.to_csv(ctx.quality_dir / "kaznerd_overlap_with_train.csv", index=False)

    ctx.r["quality"] = {"summary": summary, "gaps": gaps, "conflicts": conflicts,
                        "ud_format": ud_format, "leaked": leaked}
    return summary


# ------------------------------------------------------------- 8.2–8.3

def step_prepare(ctx, seed=42):
    """Табл. 8, этапы 2–3: единый формат, кодирование меток, очистка, разбиения."""
    manifest = prepare(ctx.raw, ctx.processed, seed=seed)
    ctx.r["manifest"] = manifest
    ctx.r["kaznerd"] = load_processed(ctx.processed, "kaznerd")
    ctx.r["udktb"] = load_processed(ctx.processed, "udktb")
    log = pd.read_json(ctx.processed / "cleaning_log.jsonl", lines=True)
    ctx.r["cleaning_log"] = log
    return manifest


# ------------------------------------------------------------- 8.4–8.5

def step_analyse(ctx):
    """Табл. 8, этапы 4–5: агрегирование по группам и описательная статистика."""
    kaz, ud = ctx.r["kaznerd"], ctx.r["udktb"]
    ud_off = ctx.r["raw"]["udktb"]

    ctx.save_table("split_sizes", eda.split_sizes(
        {"KazNERD": kaz, "UD-KTB official": ud_off, "UD-KTB re-split": ud}), index=False)
    ner = ctx.save_table("ner_entities_by_type", eda.ner_class_counts(kaz).rename_axis("entity_type"))
    ctx.save_table("ner_split_shares", eda.split_consistency(ner))
    upos = ctx.save_table("upos_by_tag", eda.token_label_counts(ud, "upos").rename_axis("upos"))
    ctx.save_table("morph_categories", eda.feature_counts(ud).to_frame("words").rename_axis("category"))
    ctx.save_table("ud_source_mix", pd.concat(
        {"official": eda.ud_domain_mix(ud_off), "re-split": eda.ud_domain_mix(ud)}, axis=1).rename_axis("source_text"))
    lengths = eda.sentence_lengths({"KazNERD": kaz, "UD-KTB": ud})
    ctx.r["lengths"] = lengths
    ctx.save_table("sentence_length_outliers", eda.length_outliers(lengths))
    ctx.save_table("vocabulary_kaznerd", eda.vocabulary_stats(kaz))
    ctx.save_table("vocabulary_udktb", eda.vocabulary_stats(ud))
    ctx.save_table("frequency_bands_kaznerd", eda.frequency_buckets(kaz))
    ctx.save_table("frequency_bands_udktb", eda.frequency_buckets(ud))
    ctx.save_table("ud_annotation_gaps", eda.ud_annotation_gaps(ud).drop(columns="by_upos"))
    ctx.save_table("token_outliers_kaznerd", eda.token_outliers(kaz))
    ctx.save_table("morph_label_space", eda.feats_label_space(ud).to_frame("value").rename_axis("measure"))

    imb = pd.concat({"KazNERD entity types": eda.imbalance_summary(ner["train"]),
                     "UD-KTB UPOS": eda.imbalance_summary(upos["train"])}, axis=1).rename_axis("measure")
    ctx.save_table("class_imbalance", imb)
    ctx.save_table("ner_class_weights", pd.DataFrame({
        "train_entities": ner["train"],
        "weight_inverse_sqrt": pd.Series(class_weights(ner["train"].to_dict(), "inverse_sqrt")),
        "weight_inverse": pd.Series(class_weights(ner["train"].to_dict(), "inverse"))}).rename_axis("entity_type"))

    desc = lengths.groupby("dataset")["length"].describe(percentiles=[.25, .5, .75, .95])
    desc["class_entropy_bits"] = [entropy_bits(ner["train"].to_dict()),
                                  entropy_bits(upos["train"].to_dict())]
    return ctx.save_table("descriptive_statistics", desc)


# ------------------------------------------------------------- 8.6

def step_visualise(ctx):
    """Табл. 8, этап 6: рисунки (PNG, 300 dpi) — данные до и после предобработки, EDA."""
    plots.style()
    t, f = ctx.r["tables"], ctx.figures
    figs = {}

    def keep(name, fig, caption):
        plots.save(fig, name, f)
        figs[name] = caption
        matplotlib.pyplot.close(fig)

    log = ctx.r["cleaning_log"]
    keep("01_cleaning_corrections", plots.cleaning_summary(log["rule"].value_counts().rename({
        "homoglyph": "Latin look-alike letters → Cyrillic", "iob2_repair": "Invalid I- tag → B- tag",
        "unicode_nfc": "Unicode NFC normalisation"})), "Corrections applied during preprocessing")
    keep("02_split_sizes", plots.split_sizes(t["split_sizes"]),
         "Sentences per split: KazNERD, UD-KTB official split and UD-KTB re-split")
    mix = t["ud_source_mix"]
    keep("03_ud_source_mix", plots.domain_mix({
        "Official train": mix[("official", "train")], "Official test": mix[("official", "test")],
        "Re-split train": mix[("re-split", "train")], "Re-split dev": mix[("re-split", "dev")],
        "Re-split test": mix[("re-split", "test")]}),
        "UD-KTB source texts by split, before and after re-splitting")
    keep("04_ner_entity_types", plots.class_distribution(
        t["ner_entities_by_type"]["train"], "KazNERD: entities per type (train)",
        "Entities (log scale)", log=True), "KazNERD entities per type in train (log scale)")
    keep("05_ner_split_consistency", plots.split_shares(
        t["ner_split_shares"], "KazNERD: entity type shares by split"),
        "Share of each entity type in train, dev and test")
    keep("06_upos_tags", plots.class_distribution(
        t["upos_by_tag"]["train"], "UD-KTB: UPOS tags (train)", "Words"), "UD-KTB UPOS tags in train")
    keep("07_morph_categories", plots.class_distribution(
        t["morph_categories"]["words"], "UD-KTB: morphological categories (train)",
        "Words carrying the feature"), "UD-KTB morphological categories in train")
    keep("08_sentence_lengths", plots.sentence_lengths(ctx.r["lengths"], t["sentence_length_outliers"]),
         "Sentence length distribution with median and IQR outlier fence")
    keep("09_word_frequency", plots.frequency_buckets({
        "KazNERD": t["frequency_bands_kaznerd"], "UD-KTB": t["frequency_bands_udktb"]}),
        "Word types and running words by frequency band in train")
    keep("10_oov_rate", plots.oov_rates({
        "KazNERD": t["vocabulary_kaznerd"], "UD-KTB": t["vocabulary_udktb"]}),
        "Share of running words in dev and test unseen in train")
    ctx.r["figures"] = figs
    return figs


# ------------------------------------------------------------- 8.8

def _pct(x):
    return f"{100 * x:.1f}%"


def step_report(ctx):
    """Табл. 8, этап 8: отчёт со всеми цифрами, собранный из результатов шагов."""
    m, t, q = ctx.r["manifest"], ctx.r["tables"], ctx.r["quality"]
    figs = ctx.r["figures"]
    fig = lambda name: rep.figure(ctx.figures / f"{name}.png", figs[name])
    kz_test_leak = m["quality"]["kaznerd"]["leakage_from_train"]["test"]["share"]
    voc_k, voc_u = t["vocabulary_kaznerd"], t["vocabulary_udktb"]
    imb = t["class_imbalance"]["KazNERD entity types"]

    rep = Report("Stage 1 data report: KazNERD and UD Kazakh-KTB")
    rep.p(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by src/pipeline.py; "
          f"preprocessing version {m['preprocessing_version']}, code commit {m['code_commit'] or 'n/a'}. "
          "Sections follow the AITU guidelines, Tables 7 and 8.")

    rep.h("1. Data collection (Table 7, step 5)")
    rep.table(pd.DataFrame([{"dataset": s["name"], "version": s["version"], "licence": s["license"],
                             "repository": s["repo"], "matches pinned version":
                             m["sources"][k]["matches_pinned_version"]} for k, s in SOURCES.items()]),
              index=False)
    if "collect" in ctx.r:                 # шаг 7.5 мог не запускаться (например, в тестах)
        rep.table(ctx.r["collect"], index=False)

    rep.h("2. Data quality check (Table 8, stage 1)")
    rep.p("Checks on the raw files before any change. Unambiguous errors are corrected in stage 3; "
          "everything else is reported and left unchanged so that official splits stay comparable.")
    rep.table(q["summary"], floatfmt="{:,.0f}")
    rep.p(f"{len(q['gaps'])} candidate missing entity labels and {len(q['conflicts'])} sentences with "
          "conflicting labels are listed in quality/ for manual review.")

    rep.h("3. Coding, structuring and transformation (Table 8, stages 2–3)")
    rep.p("Both corpora are converted to one JSONL schema (id, tokens, labels per token). UD multiword "
          f"tokens ({q['ud_format']['multiword_tokens']}) are represented by their syntactic words. "
          "Label maps are built from train only and shared by all methods.")
    rep.table(pd.Series(m["label_space"]).rename_axis("label set").to_frame("labels"), floatfmt="{:,.0f}")
    rep.table(pd.Series(m["cleaning"]).rename_axis("rule").to_frame("corrections"), floatfmt="{:,.0f}")
    fig("01_cleaning_corrections")
    rep.p("KazNERD keeps its official split. The official UD-KTB train set has only "
          f"{m['counts']['udktb_official']['train']['sentences']} sentences from different source texts "
          "than test, so train and test were pooled and re-split 80/10/10, stratified by source text, seed "
          f"{m['params']['seed']}.")
    rep.table(t["split_sizes"], index=False)
    fig("02_split_sizes")
    fig("03_ud_source_mix")

    rep.h("4. Aggregation and statistical analysis (Table 8, stages 4–5)")
    rep.p(f"Entity types are highly imbalanced: {imb['largest_class']} alone is {_pct(imb['largest_share'])} "
          f"of entities, the three largest types {_pct(imb['top3_share'])}, and {imb['smallest_class']} has "
          f"{int(imb['smallest_count'])} training entities (ratio {imb['imbalance_ratio']:,.0f}:1). "
          "Macro-F1 and per-class F1 are therefore reported alongside micro-F1.")
    rep.table(t["ner_entities_by_type"], floatfmt="{:,.0f}")
    fig("04_ner_entity_types")
    fig("05_ner_split_consistency")
    rep.table(t["class_imbalance"])
    fig("06_upos_tags")
    fig("07_morph_categories")
    rep.table(t["morph_label_space"])
    rep.table(t["descriptive_statistics"], floatfmt="{:,.2f}")
    fig("08_sentence_lengths")
    rep.p(f"Out-of-vocabulary rate (running words of test unseen in train): KazNERD "
          f"{_pct(voc_k.loc['test', 'oov_token_rate'])}, UD-KTB {_pct(voc_u.loc['test', 'oov_token_rate'])}. "
          "For the agglutinative Kazakh morphology with a small training set this is the main difficulty "
          "and the motivation for morphological rules in the hybrid.")
    rep.table(pd.concat({"KazNERD": voc_k, "UD-KTB": voc_u}).rename_axis(["dataset", "split"]))
    fig("09_word_frequency")
    fig("10_oov_rate")
    rep.p(f"{_pct(kz_test_leak)} of KazNERD test sentences also occur verbatim in train.")

    rep.h("5. Confidentiality and documentation (Table 8, stages 9–10)")
    rep.p("No personal data is collected; both corpora are public and used under their licences. "
          "Raw and processed data are stored on Google Drive, not in git. SHA-256 hashes of all inputs "
          "and outputs, the seed and the code commit are recorded in processed/manifest.json; every "
          "correction is listed in processed/cleaning_log.jsonl.")

    rep.to_markdown(ctx.out / "report.md")
    rep.to_html(ctx.out / "report.html")
    return ctx.out / "report.html"


# ------------------------------------------------------------- 8.9–8.10

def step_document(ctx, timings):
    """Табл. 8, этапы 9–10: журнал запуска — что, когда, с какими версиями и лицензиями."""
    log = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": ctx.r["manifest"]["code_commit"],
        "steps": timings,
        "sources": {k: {"version": s["version"], "license": s["license"], "citation": s["citation"]}
                    for k, s in SOURCES.items()},
        "storage": {"raw_and_processed_data": "Google Drive (not in git)",
                    "code": "private GitHub repository",
                    "credentials": "GitHub token in Colab Secrets, never in code"},
        "outputs": sorted(str(p.relative_to(ctx.out)) for p in ctx.out.rglob("*") if p.is_file()),
    }
    with open(ctx.out / "pipeline_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    return log


# ------------------------------------------------------------- запуск

STEPS = [
    ("7.5", "Data collection", lambda ctx, a: step_collect(ctx, a.download)),
    ("8.1", "Data quality check", lambda ctx, a: step_quality(ctx)),
    ("8.2–8.3", "Coding, structuring, transformation", lambda ctx, a: step_prepare(ctx, a.seed)),
    ("8.4–8.5", "Aggregation and statistical analysis", lambda ctx, a: step_analyse(ctx)),
    ("8.6", "Data visualisation", lambda ctx, a: step_visualise(ctx)),
    ("8.8", "Creating a report", lambda ctx, a: step_report(ctx)),
]


def run(thesis_dir, download=False, seed=42, verbose=True):
    args = argparse.Namespace(download=download, seed=seed)
    ctx = Context(thesis_dir)
    timings = []
    for ref, name, fn in STEPS:
        t0 = time.time()
        fn(ctx, args)
        timings.append({"guideline_step": ref, "name": name, "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"  [{ref:>7}] {name:<40} {timings[-1]['seconds']:6.1f} s")
    step_document(ctx, timings)
    if verbose:
        print(f"  [{'8.9–8.10':>7}] {'Confidentiality and documentation':<40}")
        print(f"\nГотово. Отчёт: {ctx.out / 'report.html'}")
    return ctx


def main():
    p = argparse.ArgumentParser(description="Этап 1: данные (по шагам методички)")
    p.add_argument("--thesis", required=True, help="папка thesis на Drive")
    p.add_argument("--download", action="store_true", help="скачать недостающие исходные файлы")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(a.thesis, a.download, a.seed)


if __name__ == "__main__":
    main()
