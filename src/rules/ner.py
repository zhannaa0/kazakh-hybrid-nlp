"""Rule-based NER: шаблоны для числовых сущностей + газеттир с фильтром надёжности.

1. Шаблоны (написаны вручную, высокая точность):
   число (цифры или числительные словами) + следующее слово определяют тип:
     «%», «пайыз», «процент»          → PERCENTAGE   (30 %-ке, жүз пайыз)
     валюта (теңге, доллар…)          → MONEY        (бес жүз мың теңге)
     единица измерения (шақырым…)     → QUANTITY     (20 тонна)
     «жыл…» после числа, месяц        → DATE         (2020 жылы, 5 мамыр)
     порядковое число                 → ORDINAL      (1-інші, екіншіден)
     иначе (цифры / составное число)  → CARDINAL
2. Газеттир из train: строки, которые в train были сущностями, с их самым частым
   типом. Чтобы не размечать «бір» или «Ауыл» всюду, строка берётся, только если
   в train она размечена как сущность не реже, чем в min_rate случаев.
   Последнее слово сравнивается по основе, если основа — известная сущность:
   «Қазақстанның» находит «Қазақстан».
Перекрытия: сначала шаблоны, потом газеттир, самые длинные совпадения первыми.
"""

import re
from collections import Counter, defaultdict

from .lexicon import CARDINAL_WORDS, CURRENCIES, MONTHS, PERCENT_WORDS, UNITS
from .morphology import MorphAnalyzer

DIGITS = re.compile(r"^\d+([.,]\d+)*(-\w+)?$")
DIGIT_ORD = re.compile(r"^\d+-?(ыншы|інші|ншы|нші|шы|ші)\w*$")
_NUM_STEMS = sorted((w for w in CARDINAL_WORDS if w not in ("млн", "млрд", "трлн", "жарым")), key=len, reverse=True)
WORD_ORD = re.compile(r"^(" + "|".join(_NUM_STEMS) + r")(ыншы|інші|ншы|нші)\w*$")
MAX_LEN = 8


class RuleNER:
    def __init__(self, train_sentences=(), min_rate=0.5, min_count=2):
        self.morph = MorphAnalyzer()
        self.closed = {w for w, an in self.morph.closed.items() if all(u != "NUM" for u, _ in an)}
        self.min_rate, self.min_count = min_rate, min_count
        self.gazetteer = {}            # ключ → (тип, доля «как сущность», число вхождений)
        self._known, self._first = set(), {}
        if train_sentences:
            self._build_gazetteer(train_sentences)

    # ------------------------------------------------------------ ключи
    def _stem(self, token):
        return self.morph.nominal_stem(token.lower()).lower()

    def _canonical(self, tokens, known):
        """Ключ строки для газеттира. Последнее слово заменяется основой, только если
        получившаяся строка — известная сущность: «Қазақстанның» → «қазақстан»,
        но «Қазақстан» не превращается в «қазақс»."""
        toks = tuple(t.lower() for t in tokens)
        for stem in self.morph.nominal_stems(toks[-1]):          # 1) известная основа (длинная первой)
            key = toks[:-1] + (stem.lower(),)
            if key in known:
                return key
        if toks in known:                                         # 2) сама форма известна
            return toks
        return toks[:-1] + (self._stem(toks[-1]),)                # 3) грубая основа — общий ключ для
                                                                  #    всех падежных форм слова

    # -------------------------------------------------------- газеттир
    def _build_gazetteer(self, train):
        from ..eval.metrics import ner_entities
        occurrences = []
        for s in train:
            for etype, a, b in ner_entities(s["ner"]):
                if b - a <= MAX_LEN:
                    occurrences.append((tuple(t.lower() for t in s["tokens"][a:b]), etype))
        known = {k for k, _ in occurrences}
        types = defaultdict(Counter)
        for k, etype in occurrences:
            types[self._canonical(k, known)][etype] += 1
        self._known = known
        self._first = defaultdict(set)              # первое слово → длины многословных ключей (ускоряет поиск)
        for k in types:
            self._first[k[0]].add(len(k))
        # сколько раз та же строка встретилась БЕЗ разметки
        as_o = Counter()
        for s in train:
            toks, tags = s["tokens"], s["ner"]
            for i, t in enumerate(toks):
                for n in self._first.get(t.lower(), set()) | {1}:     # однословные — по основе, проверяем всегда
                    if i + n <= len(toks) and all(x == "O" for x in tags[i:i + n]):
                        key = self._canonical(toks[i:i + n], known)
                        if key in types:
                            as_o[key] += 1
        for key, c in types.items():
            n_ent = sum(c.values())
            rate = n_ent / (n_ent + as_o[key])
            if n_ent >= self.min_count and rate >= self.min_rate:
                self.gazetteer[key] = (c.most_common(1)[0][0], rate, n_ent)

    # --------------------------------------------------------- шаблоны
    def _is_number(self, tok):
        low = tok.lower()
        if DIGITS.match(low) or low in CARDINAL_WORDS:
            return True
        # «оннан», «жүзден» — числительное с аффиксом; но не «бірге», «бірден» (наречия)
        return self._stem(low) in CARDINAL_WORDS and low not in self.closed

    def _is_ordinal(self, tok):
        """1-інші, 2-ншісі, бірінші, екіншіден: число + порядковый аффикс (+ другие аффиксы)."""
        low = tok.lower()
        return bool(DIGIT_ORD.match(low) or WORD_ORD.match(low))

    def _pattern_spans(self, tokens):
        spans, i, n = [], 0, len(tokens)
        while i < n:
            if self._is_ordinal(tokens[i]):
                spans.append(("ORDINAL", i, i + 1, 0.9))
                i += 1
                continue
            if not self._is_number(tokens[i]):
                low = tokens[i].lower()
                if self._stem(low) in MONTHS or low in MONTHS:
                    spans.append(("DATE", i, i + 1, 0.8))
                i += 1
                continue
            j = i
            while j < n and self._is_number(tokens[j]):
                j += 1
            if j < n and self._is_ordinal(tokens[j]) and j + 1 < n and tokens[j + 1].lower().startswith("жыл"):
                j += 1                                         # екі мың оныншы жылы → DATE
            nxt = tokens[j].lower() if j < n else ""
            nxt_stem = self._stem(nxt) if nxt else ""
            multiword = j - i > 1
            if nxt.startswith("%") or nxt_stem in PERCENT_WORDS or nxt in PERCENT_WORDS:
                spans.append(("PERCENTAGE", i, j + 1, 0.95))
            elif nxt_stem in CURRENCIES or nxt in CURRENCIES:
                spans.append(("MONEY", i, j + 1, 0.95))
            elif nxt_stem in UNITS or nxt in UNITS:
                spans.append(("QUANTITY", i, j + 1, 0.85))
            elif nxt.startswith("жыл") and not nxt.startswith("жылқы"):
                spans.append(("DATE", i, j + 1, 0.9))
            elif nxt_stem in MONTHS or nxt in MONTHS:
                end = j + 1
                if end + 1 < n and DIGITS.match(tokens[end]) and tokens[end + 1].lower().startswith("жыл"):
                    end += 2                                   # 5 мамыр 2020 жылы
                spans.append(("DATE", i, end, 0.9))
            elif DIGITS.match(tokens[i].lower()) or multiword:
                spans.append(("CARDINAL", i, j, 0.7))
            else:
                spans.append(None)                             # одно числительное словом — решит газеттир
                spans.pop()
                i += 1
                continue
            i = spans[-1][2]
        return spans

    # ------------------------------------------------------------ разметка
    def spans(self, tokens):
        """Список (тип, начало, конец, уверенность, источник) без перекрытий."""
        taken = [False] * len(tokens)
        out = []
        for etype, a, b, conf in self._pattern_spans(tokens):
            out.append((etype, a, b, conf, "pattern"))
            for k in range(a, b):
                taken[k] = True
        i = 0
        while i < len(tokens):
            if taken[i]:
                i += 1
                continue
            for L in range(min(MAX_LEN, len(tokens) - i), 0, -1):
                if any(taken[i:i + L]):
                    continue
                if not self.gazetteer or (L > 1 and L not in self._first.get(tokens[i].lower(), ())):
                    continue
                hit = self.gazetteer.get(self._canonical(tokens[i:i + L], self._known))
                # служебное слово («бірге») — сущность, только если в train оно почти всегда размечено
                # («бүгін», «биыл» в KazNERD — DATE)
                if hit and L == 1 and tokens[i].lower() in self.closed and hit[1] < 0.8:
                    hit = None
                if hit:
                    out.append((hit[0], i, i + L, hit[1], "gazetteer"))
                    for k in range(i, i + L):
                        taken[k] = True
                    i += L - 1
                    break
            i += 1
        return sorted(out, key=lambda x: x[1])

    def tag(self, tokens):
        tags = ["O"] * len(tokens)
        for etype, a, b, _, _ in self.spans(tokens):
            tags[a] = f"B-{etype}"
            for k in range(a + 1, b):
                tags[k] = f"I-{etype}"
        return tags
