"""Обучение и предсказание нейросетевого метода (запускается в Colab на GPU).

    # NER на KazNERD
    python -m src.neural.train --thesis /content/drive/MyDrive/thesis --task ner --seed 13
    # POS + морфология на UD-KTB одной моделью (общий трансформер, головы UPOS и FEATS)
    python -m src.neural.train --thesis /content/drive/MyDrive/thesis --task ud --seed 13

Результат — в thesis/outputs/preds/{dev,test}/:
    ner_neural_seed13.jsonl           (task=ner)
    pos_neural_seed13.jsonl, morph_neural_seed13.jsonl   (task=ud)
В каждом файле, кроме pred, поле "probs" — вероятности модели для каждого слова;
его использует гибрид. Журнал обучения — thesis/outputs/neural_logs/.

Лучшая эпоха выбирается по dev (NER: F1; UD: среднее accuracy UPOS и exact match FEATS);
test используется только для итогового предсказания.
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..data.encoding import columns_to_feats, load_label_maps
from ..data.prepare import load_processed
from ..eval.metrics import morph_metrics, ner_metrics, pos_metrics
from .data import collate, encode, head_specs
from .model import MultiHeadTagger

DEFAULTS = {
    "ner": {"epochs": 3, "lr": 3e-5, "batch": 16, "patience": 1},
    "ud": {"epochs": 40, "lr": 5e-5, "batch": 16, "patience": 6},
}
TOP_K = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(name):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, add_prefix_space=True)   # нужно RoBERTa-BPE для слов
    except TypeError:
        return AutoTokenizer.from_pretrained(name)


def build_model(name, heads, class_weights=None):
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(name)
    return MultiHeadTagger(backbone, {k: len(m) for k, m in heads.items()}, class_weights=class_weights)


# ------------------------------------------------------------- предсказание

@torch.no_grad()
def predict(model, examples, heads, device, batch_size=32, pad_id=1):
    """Для каждого предложения: {голова: [распределение вероятностей по словам]}."""
    model.eval()
    out = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        ids, mask, _ = collate(batch, pad_id)
        logits, _ = model(ids.to(device), mask.to(device))
        probs = {k: torch.softmax(v.float(), -1).cpu().numpy() for k, v in logits.items()}
        for j, e in enumerate(batch):
            first = {}
            for pos, w in enumerate(e["word_ids"]):
                if w is not None and w not in first:
                    first[w] = pos
            per_head = {}
            for k in heads:
                rows = [probs[k][j, first[w]] if w in first else None for w in range(e["n_words"])]
                per_head[k] = rows
            out.append(per_head)
    return out


FALLBACK = {"ner": "O", "upos": "NOUN", "feats": "_"}


def _argmax_label(m, row, fallback):
    return fallback if row is None else m.labels[int(row.argmax())]


def _topk(m, row, k=TOP_K):
    if row is None:
        return []
    idx = np.argsort(-row)[:k]
    return [[m.labels[int(i)], round(float(row[i]), 5)] for i in idx]


def to_records(sentences, outputs, heads, task):
    """Распределения → файлы предсказаний в формате модуля оценки (+ probs для гибрида)."""
    if task == "ner":
        m = heads["ner"]
        return {"ner": [{"id": s["id"], "tokens": s["tokens"], "gold": s["ner"],
                         "pred": [_argmax_label(m, r, "O") for r in o["ner"]],
                         "probs": [_topk(m, r) for r in o["ner"]]} for s, o in zip(sentences, outputs)]}
    pos, morph = [], []
    cats = {name[5:]: m for name, m in heads.items() if name.startswith("feat:")}
    for s, o in zip(sentences, outputs):
        um = heads["upos"]
        pos.append({"id": s["id"], "tokens": s["tokens"], "gold": s["upos"],
                    "pred": [_argmax_label(um, r, "NOUN") for r in o["upos"]],
                    "probs": [_topk(um, r) for r in o["upos"]]})
        if "feats" in heads:
            fm = heads["feats"]
            pred = [_argmax_label(fm, r, "_") for r in o["feats"]]
            probs = [_topk(fm, r) for r in o["feats"]]
        else:
            pred, probs = [], []
            for w in range(len(s["tokens"])):
                cols, dist = {}, {}
                for k, m in cats.items():
                    r = o[f"feat:{k}"][w]
                    cols[k] = _argmax_label(m, r, "_")
                    dist[k] = _topk(m, r, k=len(m))
                pred.append(columns_to_feats(cols))
                probs.append(dist)
        morph.append({"id": s["id"], "tokens": s["tokens"], "gold": s["feats"], "pred": pred,
                      "probs": probs, "morph_mode": "whole" if "feats" in heads else "split"})
    return {"pos": pos, "morph": morph}


def dev_score(records, task):
    if task == "ner":
        return ner_metrics(records["ner"])["f1"]
    return (pos_metrics(records["pos"])["accuracy"] + morph_metrics(records["morph"])["exact_match"]) / 2


# ------------------------------------------------------------------ обучение

def train(thesis, task, model_name="xlm-roberta-base", seed=13, epochs=None, lr=None, batch=None,
          patience=None, morph_mode="split", class_weights="none", max_train=None, max_length=256,
          device=None, verbose=True, tokenizer=None, backbone=None):
    """tokenizer / backbone можно передать готовыми (так делают тесты с крошечной моделью)."""
    from transformers import get_linear_schedule_with_warmup
    cfg = {**DEFAULTS[task], **{k: v for k, v in dict(epochs=epochs, lr=lr, batch=batch,
                                                       patience=patience).items() if v is not None}}
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    thesis = Path(thesis)
    proc = thesis / "data" / "processed"
    data = load_processed(proc, "kaznerd" if task == "ner" else "udktb")
    if max_train:
        data["train"] = data["train"][:max_train]
    maps = load_label_maps(proc / "label_maps.json")
    heads = head_specs(task, maps, morph_mode)

    weights = None
    if task == "ner" and class_weights != "none":
        from collections import Counter
        from ..data.encoding import class_weights as cw
        counts = Counter(t for s in data["train"] for t in s["ner"])
        w = cw({lab: counts.get(lab, 1) for lab in heads["ner"].labels}, class_weights)
        weights = {"ner": [w[lab] for lab in heads["ner"].labels]}

    tok = tokenizer or load_tokenizer(model_name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    enc = {sp: encode(tok, data[sp], task, heads, max_length) for sp in ("train", "dev", "test")}
    if backbone is not None:
        model = MultiHeadTagger(backbone, {k: len(m) for k, m in heads.items()}, class_weights=weights)
    else:
        model = build_model(model_name, heads, weights)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(enc["train"]) / cfg["batch"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps_per_epoch * cfg["epochs"]),
                                            steps_per_epoch * cfg["epochs"])
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(seed)

    log = {"task": task, "model": model_name, "seed": seed, "morph_mode": morph_mode,
           "class_weights": class_weights, "config": cfg, "device": device, "epochs": []}
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = list(range(len(enc["train"])))
        rng.shuffle(order)
        total = 0.0
        for b in range(0, len(order), cfg["batch"]):
            ids, mask, labels = collate([enc["train"][i] for i in order[b:b + cfg["batch"]]], pad_id)
            labels = {k: v.to(device) for k, v in labels.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(ids.to(device), mask.to(device), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        dev_rec = to_records(data["dev"], predict(model, enc["dev"], heads, device, pad_id=pad_id), heads, task)
        score = dev_score(dev_rec, task)
        log["epochs"].append({"epoch": epoch, "train_loss": total / steps_per_epoch, "dev_score": score,
                              "minutes": round((time.time() - t0) / 60, 1)})
        if verbose:
            print(f"  epoch {epoch:2d}  loss {total / steps_per_epoch:.4f}  dev {score:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad > cfg["patience"]:
                break

    model.load_state_dict(best_state)
    log["best_dev_score"] = best
    out_root = thesis / "outputs" / "preds"
    for sp in ("dev", "test"):
        recs = to_records(data[sp], predict(model, enc[sp], heads, device, pad_id=pad_id), heads, task)
        for name, rows in recs.items():
            path = out_root / sp / f"{name}_neural_seed{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log["minutes"] = round((time.time() - t0) / 60, 1)
    logs = thesis / "outputs" / "neural_logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / f"{task}_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    if verbose:
        print(f"  лучший dev {best:.4f}; предсказания в {out_root}")
    return log


def main():
    p = argparse.ArgumentParser(description="Нейросетевой метод: общий трансформер + головы задач")
    p.add_argument("--thesis", required=True)
    p.add_argument("--task", choices=["ner", "ud"], required=True)
    p.add_argument("--model", default="xlm-roberta-base",
                   help="xlm-roberta-base или kz-transformers/kaz-roberta-conversational")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch", type=int)
    p.add_argument("--morph-mode", choices=["split", "whole"], default="split")
    p.add_argument("--class-weights", choices=["none", "inverse_sqrt", "inverse"], default="none")
    p.add_argument("--max-train", type=int, help="обрезать train (для быстрой проверки)")
    a = p.parse_args()
    train(a.thesis, a.task, a.model, a.seed, a.epochs, a.lr, a.batch, None, a.morph_mode,
          a.class_weights, a.max_train)


if __name__ == "__main__":
    main()
