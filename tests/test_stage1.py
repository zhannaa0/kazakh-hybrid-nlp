"""Тесты новых шагов методички: сбор (7.5), кодирование (8.2–8.3),
статистика результатов (8.4–8.5), отчёт (8.8) и весь конвейер на синтетических данных."""

import json

import pytest

from src.data import collect as C
from src.data.encoding import (IGNORE_INDEX, LabelMap, align_to_subwords, build_label_maps,
                               class_weights, columns_to_feats, feats_to_columns, ner_labels,
                               words_from_subwords)
from src.eval.breakdown import by_length, by_oov, train_vocabulary
from src.eval.significance import holm, paired_bootstrap_runs
from src.report import Report


# ------------------------------------------------------------ 7.5 сбор

def test_verify_reports_missing_and_mismatch(tmp_path, monkeypatch):
    good = tmp_path / "ds" / "a.txt"
    good.parent.mkdir()
    good.write_text("abc", encoding="utf-8")
    monkeypatch.setattr(C, "SOURCES", {"ds": {"name": "DS", "base_url": "", "files": {
        "a.txt": C.sha256(good), "b.txt": "0" * 64}}})
    status = {r["file"]: r["status"] for r in C.verify(tmp_path)}
    assert status == {"a.txt": "ok", "b.txt": "missing"}

    (tmp_path / "ds" / "b.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HASH MISMATCH"):
        C.collect(tmp_path, download=False)


def test_pinned_sources_are_complete():
    for src in C.SOURCES.values():
        assert {"name", "repo", "version", "license", "citation", "base_url", "files"} <= src.keys()
        assert all(len(h) == 64 for h in src["files"].values())


# ------------------------------------------------------- 8.2–8.3 кодирование

def test_label_map_roundtrip_and_unknown():
    m = LabelMap(["O", "B-PER", "I-PER", "O"])
    assert len(m) == 3
    assert m.decode(m.encode(["B-PER", "O"])) == ["B-PER", "O"]
    assert m.encode(["B-XYZ"], unknown=0) == [0]
    with pytest.raises(KeyError):
        m.encode(["B-XYZ"])


def test_ner_labels_full_iob2_scheme():
    train = [{"ner": ["B-PER", "I-PER", "O", "B-GPE"]}]
    assert ner_labels(train) == ["O", "B-GPE", "I-GPE", "B-PER", "I-PER"]


def test_feats_columns_roundtrip():
    ud_train = [{"upos": ["NOUN", "VERB"], "feats": ["Case=Nom|Number=Sing", "Tense=Past"]}]
    maps = build_label_maps([{"ner": ["O"]}], ud_train)
    for f in ("Case=Nom|Number=Sing", "Tense=Past", "_"):
        assert columns_to_feats(feats_to_columns(f, maps["feat_values"])) == f
    assert maps["feat_values"]["Case"] == ["_", "Nom"]


def test_class_weights_rare_class_heavier_and_capped():
    w = class_weights({"big": 10_000, "small": 100, "tiny": 1}, "inverse", max_weight=2)
    assert w["big"] < w["small"] < w["tiny"] == 2
    assert class_weights({"a": 5, "b": 50}, "none") == {"a": 1.0, "b": 1.0}


def test_subword_alignment_roundtrip():
    word_ids = [None, 0, 0, 1, 2, 2, 2, None]
    aligned = align_to_subwords(word_ids, [5, 7, 9])
    assert aligned == [IGNORE_INDEX, 5, IGNORE_INDEX, 7, 9, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert words_from_subwords(word_ids, [0, 5, 1, 7, 9, 2, 2, 0]) == [5, 7, 9]


# ---------------------------------------------------- 8.4–8.5 статистика

def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


def _sents(preds, gold=("NOUN", "VERB", "ADJ", "PUNCT"), n=60):
    return [{"id": i, "tokens": ["a", "b", "c", "."], "gold": list(gold), "pred": list(preds)}
            for i in range(n)]


def test_bootstrap_over_seeds():
    gold = ["NOUN", "VERB", "ADJ", "PUNCT"]
    weak = [_sents(["NOUN", "NOUN", "NOUN", "PUNCT"]), _sents(["NOUN", "VERB", "NOUN", "PUNCT"])]
    strong = [_sents(gold), _sents(gold)]
    r = paired_bootstrap_runs(weak, strong, "pos", n_samples=300)
    assert r["score_a"] == pytest.approx(0.625) and r["score_b"] == 1.0
    assert r["p_value"] < 0.05 and r["n_runs_a"] == 2


def test_breakdown_by_oov_and_length():
    vocab = train_vocabulary([{"tokens": ["Астана", "қаласы"]}])
    sents = [{"id": 0, "tokens": ["Астана", "жаңа"], "gold": ["PROPN", "ADJ"], "pred": ["PROPN", "NOUN"]}]
    t = by_oov(sents, "pos", vocab)
    assert t.loc["in_vocab", "score"] == 1.0 and t.loc["oov", "score"] == 0.0

    ner = [{"id": 0, "tokens": ["Астана", "Алматы", "."], "gold": ["B-GPE", "B-GPE", "O"],
            "pred": ["B-GPE", "O", "O"]}]
    t = by_oov(ner, "ner", vocab)
    assert t.loc["in_vocab", "f1"] == 1.0 and t.loc["oov", "recall"] == 0.0
    assert by_length(sents, "pos").loc["1–10", "score"] == 0.5


# ------------------------------------------------------------ 8.8 отчёт

def test_report_markdown_and_html(tmp_path):
    import pandas as pd
    rep = Report("Test report")
    rep.h("Section")
    rep.p("Text")
    rep.table(pd.DataFrame({"n": [1234], "share": [0.5]}, index=["a"]))
    rep.to_markdown(tmp_path / "r.md")
    rep.to_html(tmp_path / "r.html")
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## Section" in md and "1,234" in md and "0.500" in md
    assert "<table" in (tmp_path / "r.html").read_text(encoding="utf-8")


# ------------------------------------------------ весь конвейер этапа 1

def _write_synthetic(raw):
    (raw / "kaznerd").mkdir(parents=True)
    (raw / "ud-ktb").mkdir()
    words = ["Астана", "Алматы", "Тоқаев", "жаңа", "үй", "салды", "Aстана", "бүгін"]
    ner = {"Астана": "B-GPE", "Алматы": "B-GPE", "Тоқаев": "B-PERSON", "Aстана": "B-GPE"}
    for k, fn in enumerate(("IOB2_train.txt", "IOB2_valid.txt", "IOB2_test.txt")):
        lines = []
        for i in range(20 if k == 0 else 6):
            toks = [words[(i + j + k) % len(words)] for j in range(3 + i % 4)] + ["."]
            lines += [f"{t} {ner.get(t, 'O')}" for t in toks] + [""]
        (raw / "kaznerd" / fn).write_text("\n".join(lines), encoding="utf-8")
    feats = {"Астана": ("PROPN", "Case=Nom"), "үй": ("NOUN", "Case=Nom|Number=Sing"),
             "салды": ("VERB", "Tense=Past"), ".": ("PUNCT", "_")}
    for k, fn in enumerate(("kk_ktb-ud-train.conllu", "kk_ktb-ud-test.conllu")):
        out = []
        for i in range(12):
            toks = ["Астана", "үй" if (i + k) % 2 else "салды", "."]
            out += [f"# sent_id = story.tagged.txt:{k}:{i}", f"# text = {' '.join(toks)}"]
            for j, t in enumerate(toks, 1):
                u, f = feats[t]
                out.append(f"{j}\t{t}\t{t}\t{u}\t_\t{f}\t0\troot\t_\t_")
            out.append("")
        (raw / "ud-ktb" / fn).write_text("\n".join(out) + "\n", encoding="utf-8")


def test_pipeline_end_to_end(tmp_path):
    from src import pipeline as P
    _write_synthetic(tmp_path / "data")
    ctx = P.Context(tmp_path)
    P.step_quality(ctx)
    P.step_prepare(ctx)
    P.step_analyse(ctx)
    P.step_visualise(ctx)
    P.step_report(ctx)
    P.step_document(ctx, timings=[])
    out = tmp_path / "outputs" / "stage1_data"
    assert (out / "report.html").exists() and (out / "report.md").exists()
    assert len(list((out / "figures").glob("*.png"))) == 10
    assert (out / "quality" / "summary.csv").exists()
    maps = json.loads((tmp_path / "data" / "processed" / "label_maps.json").read_text(encoding="utf-8"))
    assert maps["ner"][0] == "O" and "B-GPE" in maps["ner"]
    assert ctx.r["manifest"]["cleaning"]["homoglyph"] > 0          # «Aстана» исправлена
