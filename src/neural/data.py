"""Подготовка данных для трансформера: слова → подслова, метки → числа на первых подсловах."""

import torch

from ..data.encoding import (IGNORE_INDEX, align_to_subwords, feats_to_columns)


def head_specs(task, maps, morph_mode="split"):
    """Какие головы у модели. task: 'ner' (KazNERD) или 'ud' (UPOS + FEATS по UD-KTB).

    morph_mode: 'split' — голова на каждую категорию (Case, Number, …), 'whole' — вся строка FEATS.
    Возвращает {имя_головы: LabelMap}.
    """
    if task == "ner":
        return {"ner": maps["ner"]}
    heads = {"upos": maps["upos"]}
    if morph_mode == "whole":
        heads["feats"] = maps["feats_whole"]
    else:
        heads.update({f"feat:{k}": m for k, m in maps["feat_values"].items()})
    return heads


def word_labels(sentence, task, heads):
    """Метки слов предложения для каждой головы (числа; неизвестная метка → IGNORE)."""
    out = {}
    if task == "ner":
        out["ner"] = heads["ner"].encode(sentence["ner"], unknown=IGNORE_INDEX)
        return out
    out["upos"] = heads["upos"].encode(sentence["upos"], unknown=IGNORE_INDEX)
    if "feats" in heads:
        out["feats"] = heads["feats"].encode(sentence["feats"], unknown=IGNORE_INDEX)
    else:
        cats = {name[5:]: m for name, m in heads.items() if name.startswith("feat:")}
        cols = [feats_to_columns(f, {k: m.labels for k, m in cats.items()}) for f in sentence["feats"]]
        for k, m in cats.items():
            out[f"feat:{k}"] = m.encode([c[k] for c in cols], unknown=IGNORE_INDEX)
    return out


def encode(tokenizer, sentences, task, heads, max_length=256):
    """Список примеров: input_ids, word_ids и выровненные метки по головам."""
    examples = []
    for s in sentences:
        enc = tokenizer(s["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
        wids = enc.word_ids()
        labels = {name: align_to_subwords(wids, ids) for name, ids in word_labels(s, task, heads).items()}
        examples.append({"input_ids": enc["input_ids"], "word_ids": wids, "labels": labels,
                         "n_words": len(s["tokens"])})
    return examples


def collate(batch, pad_id):
    n = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    labels = {name: torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long) for name in batch[0]["labels"]}
    for i, e in enumerate(batch):
        L = len(e["input_ids"])
        ids[i, :L] = torch.tensor(e["input_ids"])
        mask[i, :L] = 1
        for name, lab in e["labels"].items():
            labels[name][i, :L] = torch.tensor(lab)
    return ids, mask, labels
