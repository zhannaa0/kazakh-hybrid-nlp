"""Тесты модуля оценки. Запуск: pytest -q"""

import pytest
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

from src.eval.evaluate import evaluate_dir
from src.eval.io import parse_name, read_predictions, write_predictions
from src.eval.metrics import (ner_metrics, morph_metrics, pos_metrics,
                              score_from_stats, sentence_stats)
from src.eval.significance import paired_bootstrap


def sent(gold, pred, sid=0):
    return {"id": sid, "tokens": [f"t{i}" for i in range(len(gold))], "gold": gold, "pred": pred}


# ---------------- NER

def test_ner_perfect():
    s = [sent(["B-PERSON", "I-PERSON", "O", "B-GPE"], ["B-PERSON", "I-PERSON", "O", "B-GPE"])]
    assert ner_metrics(s)["f1"] == 1.0


def test_ner_partial_entity_is_wrong():
    # Нашли только половину имени — в strict-режиме это ошибка, а не «полупопадание»
    s = [sent(["B-PERSON", "I-PERSON", "O"], ["B-PERSON", "O", "O"])]
    m = ner_metrics(s)
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_ner_bootstrap_stats_match_seqeval():
    sents = [
        sent(["B-PERSON", "I-PERSON", "O", "B-GPE"], ["B-PERSON", "I-PERSON", "O", "O"], 0),
        sent(["O", "B-ORG", "I-ORG"], ["O", "B-ORG", "I-ORG"], 1),
        sent(["B-GPE", "O"], ["B-PERSON", "O"], 2),
        sent(["O", "O"], ["I-GPE", "O"], 3),            # некорректный I- без B-
    ]
    stats = sentence_stats(sents, "ner")
    a, b, c = (sum(x[k] for x in stats) for k in range(3))
    expected = f1_score([s["gold"] for s in sents], [s["pred"] for s in sents],
                        mode="strict", scheme=IOB2)
    assert score_from_stats(a, b, c, "ner") == pytest.approx(expected)


# ---------------- POS

def test_pos_accuracy():
    s = [sent(["NOUN", "VERB", "PUNCT", "ADJ"], ["NOUN", "NOUN", "PUNCT", "ADJ"])]
    assert pos_metrics(s)["accuracy"] == 0.75


# ---------------- MORPH

def test_morph_order_of_features_does_not_matter():
    s = [sent(["Case=Nom|Number=Sing"], ["Number=Sing|Case=Nom"])]
    assert morph_metrics(s)["exact_match"] == 1.0


def test_morph_partial_credit():
    s = [sent(["Case=Loc|Number=Sing", "_"], ["Case=Nom|Number=Sing", "_"])]
    m = morph_metrics(s)
    assert m["exact_match"] == 0.5          # 1 из 2 токенов полностью верен
    assert m["feat_f1"] == 0.5              # Number верен, Case нет
    assert m["per_class"]["Case"]["f1"] == 0.0


# ---------------- IO и имена файлов

def test_parse_name():
    assert parse_name("pos_neural_seed13.jsonl") == ("pos", "neural", "seed13")
    assert parse_name("ner_rule.jsonl") == ("ner", "rule", "default")
    with pytest.raises(ValueError):
        parse_name("ner_crf.jsonl")


def test_length_mismatch_is_caught(tmp_path):
    path = tmp_path / "pos_rule.jsonl"
    path.write_text('{"tokens": ["а", "б"], "gold": ["X", "Y"], "pred": ["X"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="разная длина"):
        read_predictions(path)


# ---------------- значимость

def test_identical_systems_not_significant():
    s = [sent(["NOUN", "VERB"], ["NOUN", "NOUN"], i) for i in range(50)]
    r = paired_bootstrap(s, s, "pos", n_samples=500)
    assert r["diff"] == 0 and r["p_value"] == 1.0


def test_clearly_better_system_is_significant():
    gold = ["NOUN", "VERB", "ADJ", "PUNCT"]
    worse = [sent(gold, ["NOUN", "NOUN", "NOUN", "PUNCT"], i) for i in range(100)]
    better = [sent(gold, gold, i) for i in range(100)]
    assert paired_bootstrap(worse, better, "pos", n_samples=500)["p_value"] < 0.05


# ---------------- весь конвейер

def test_evaluate_dir_builds_3x3(tmp_path):
    preds = tmp_path / "preds"
    gold = ["NOUN", "VERB"]
    write_predictions(preds / "pos_rule.jsonl", [sent(gold, ["NOUN", "NOUN"])])
    write_predictions(preds / "pos_neural_seed1.jsonl", [sent(gold, gold)])
    write_predictions(preds / "pos_neural_seed2.jsonl", [sent(gold, ["NOUN", "NOUN"])])
    table, runs = evaluate_dir(preds, tmp_path)
    assert table.loc["pos", "rule"] == "50.00"
    assert table.loc["pos", "neural"].startswith("75.00 ±")
    assert (tmp_path / "summary_3x3.csv").exists()
