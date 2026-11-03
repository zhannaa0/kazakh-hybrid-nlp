"""Сквозные тесты: правила → (нейросеть) → гибрид → оценка на синтетических данных.

Нейросеть проверяется на КРОШЕЧНОЙ случайно инициализированной модели (без загрузки
из интернета) — только то, что код работает от начала до конца, а не качество.
Если torch/transformers не установлены, нейросетевой тест пропускается.
"""

import json
import random

import pytest

from src.data.prepare import prepare
from src.eval.io import read_predictions
from tests.test_stage1 import _write_synthetic


@pytest.fixture()
def thesis(tmp_path):
    _write_synthetic(tmp_path / "data")
    prepare(tmp_path / "data", tmp_path / "data" / "processed")
    return tmp_path


def _fake_neural(rule_path, out_path, task, labels, seed=0):
    """Имитация нейросети: золотая разметка с 20% ошибок и вероятностями."""
    rng = random.Random(seed)
    rows = []
    for r in read_predictions(rule_path):
        pred, probs = [], []
        for g in r["gold"]:
            y = g if rng.random() > 0.2 else rng.choice(labels)
            other = rng.choice([l for l in labels if l != y])
            pred.append(y)
            probs.append([[y, 0.7], [other, 0.3]])
        row = {"id": r["id"], "tokens": r["tokens"], "gold": r["gold"], "pred": pred, "probs": probs}
        rows.append(row)
    out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")


def test_rules_then_hybrid_then_evaluate(thesis):
    from src.eval.evaluate import compare_all, evaluate_dir
    from src.hybrid import run as hybrid
    from src.rules import run as rules

    rules.run(thesis, verbose=False)
    preds = thesis / "outputs" / "preds"
    labels = {"ner": ["O", "B-GPE", "B-PERSON"], "pos": ["NOUN", "PROPN", "VERB", "PUNCT"],
              "morph": ["Case=Nom", "_", "Tense=Past", "Case=Nom|Number=Sing"]}
    for sp in ("dev", "test"):
        for task, labs in labels.items():
            _fake_neural(preds / sp / f"{task}_rule.jsonl", preds / sp / f"{task}_neural_seed1.jsonl", task, labs)

    reports = hybrid.run(thesis, seed=1, verbose=False)
    assert set(reports) == {"ner", "pos", "morph"}
    for task, r in reports.items():
        assert r["dev_hybrid"] >= max(r["dev_neural"], 0) - 1e-9       # λ=0 / «всё выключено» в сетке
        assert (preds / "test" / f"{task}_hybrid_seed1.jsonl").exists()
    assert (thesis / "outputs" / "hybrid" / "params_seed1.json").exists()

    table, _ = evaluate_dir(preds / "test", thesis / "res")
    assert list(table.columns) == ["metric", "rule", "neural", "hybrid"]
    sig = compare_all(preds / "test", thesis / "res", n_samples=200)
    assert len(sig) == 9 and "p_holm" in sig


def test_neural_training_tiny_model(thesis):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    from src.neural.train import train

    words = []
    for sp in ("train",):
        for f in (thesis / "data" / "processed").rglob(f"{sp}.jsonl"):
            words += [t for line in f.read_text(encoding="utf-8").splitlines()
                      for t in json.loads(line)["tokens"]]
    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.train_from_iterator([" ".join(words)], trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    from tokenizers.processors import TemplateProcessing
    tk.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                                           special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                               cls_token="[CLS]", sep_token="[SEP]")

    def tiny():
        cfg = transformers.BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=64)
        return transformers.BertModel(cfg)

    for task in ("ud", "ner"):
        log = train(thesis, task, "tiny", seed=1, epochs=2, batch=8, device="cpu", verbose=False,
                    tokenizer=tok, backbone=tiny())
        assert len(log["epochs"]) >= 1
    preds = thesis / "outputs" / "preds"
    for name in ("ner", "pos", "morph"):
        rows = read_predictions(preds / "test" / f"{name}_neural_seed1.jsonl")
        assert all(len(r["probs"]) == len(r["tokens"]) for r in rows)
    morph = read_predictions(preds / "dev" / "morph_neural_seed1.jsonl")
    assert isinstance(morph[0]["probs"][0], dict)                     # FEATS по категориям

    from src.hybrid import run as hybrid
    from src.rules import run as rules
    rules.run(thesis, verbose=False)
    reports = hybrid.run(thesis, seed=1, verbose=False)
    assert set(reports) == {"ner", "pos", "morph"}
