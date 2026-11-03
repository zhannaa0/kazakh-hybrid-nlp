"""Нейросетевая модель: один общий трансформер-«хребет» + отдельные линейные головы.

    h_i = Transformer(x)_i                  — контекстное представление i-го подслова
    p(y_i^(k) | x) = softmax(W_k · h_i + b_k) — голова k (NER, UPOS, каждая категория FEATS…)

Слово получает предсказание своего первого подслова. Обучение — сумма
перекрёстных энтропий по головам (подслова, кроме первого, не учитываются):

    L = Σ_k Σ_i  w^(k)_{y_i} · ( −log p(y_i^(k) | x) )

w — веса классов (по умолчанию все 1; для NER можно взять 1/√n против дисбаланса).
"""

import torch
from torch import nn

from ..data.encoding import IGNORE_INDEX


class MultiHeadTagger(nn.Module):
    def __init__(self, backbone, heads, dropout=0.1, class_weights=None):
        """backbone — модель transformers (AutoModel); heads — {имя: число классов}."""
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, n) for name, n in heads.items()})
        self.losses = nn.ModuleDict()
        for name in heads:
            w = (class_weights or {}).get(name)
            self.losses[name] = nn.CrossEntropyLoss(
                ignore_index=IGNORE_INDEX, weight=None if w is None else torch.tensor(w, dtype=torch.float))

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        logits = {name: head(h) for name, head in self.heads.items()}
        loss = None
        if labels is not None:
            loss = sum(self.losses[name](logits[name].flatten(0, 1).float(), labels[name].flatten())
                       for name in self.heads)
        return logits, loss
