"""Общая rule-based морфологическая инфраструктура (POS + морфология + основы для NER).

Анализатор возвращает для слова МНОЖЕСТВО кандидатов-разборов с оценками:

    analyze("кітаптарымызда") → [Analysis(NOUN, Case=Loc|Number=Plur|Number[psor]=Plur|Person[psor]=1, ...), ...]

Источники кандидатов (по убыванию доверия):
  1. пунктуация и числа (регулярные выражения);
  2. словарь словоформ, собранный из train (лексикон — обычная часть rule-based систем);
  3. словарь служебных слов, составленный вручную (lexicon.py);
  4. морфотактика: перебор цепочек аффиксов «основа + мн. число + притяжательность + падеж»
     для имён и набор глагольных окончаний — правила записаны вручную по грамматике;
  5. умолчание: NOUN Case=Nom (PROPN, если слово с заглавной буквы не в начале предложения).

Контекстные правила (RuleTagger) затем меняют оценки с учётом соседей, и
rule-based метод выбирает кандидата с максимальной оценкой. Гибрид использует
ВСЁ множество кандидатов (src/hybrid).
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product

from .lexicon import closed_class_lexicon

PUNCT_RE = re.compile(r"^[^\w\s]+$|^[-–—]+$")
NUM_RE = re.compile(r"^\d+([.,]\d+)*%?$")
NUM_ORD_RE = re.compile(r"^\d+-?(ыншы|інші|ншы|нші|шы|ші)\w*$")
NUM_SUFFIX_RE = re.compile(r"^\d+([.,]\d+)*-\w+$")
VOWELS = set("аәеиоөұүыіэюяё")


@dataclass(frozen=True)
class Analysis:
    upos: str
    feats: str
    lemma: str
    source: str           # откуда кандидат: punct / number / train_lexicon / closed_class / suffix / default
    score: float = 0.0
    rule: str = ""        # какой аффикс / правило сработало


def join_feats(*parts):
    """Склеить наборы признаков 'A=1|B=2' в один, в порядке UD (по алфавиту без учёта регистра)."""
    items = {}
    for p in parts:
        if p and p != "_":
            for kv in p.split("|"):
                k, v = kv.split("=", 1)
                items[k] = v
    return "|".join(f"{k}={v}" for k, v in sorted(items.items(), key=lambda kv: kv[0].lower())) or "_"


# ================================================================ имена
# Слоты именной морфотактики: (аффиксы-алломорфы, признаки). Варианты по
# сингармонизму и ассимиляции перечислены явно — так правила легко проверить.
PLURAL = (["лар", "лер", "дар", "дер", "тар", "тер"], "Number=Plur")
POSS3 = "Number[psor]=Plur,Sing|Person[psor]=3"
POSSESSIVES = [
    (["ы", "і", "сы", "сі"], POSS3, "3"),
    (["ым", "ім", "м"], "Number[psor]=Sing|Person[psor]=1", "12"),
    (["ымыз", "іміз", "мыз", "міз"], "Number[psor]=Plur|Person[psor]=1", "12"),
    (["ың", "ің", "ң"], "Number[psor]=Sing|Person[psor]=2", "12"),
    (["ыңыз", "іңіз", "ңыз", "ңіз"], "Number[psor]=Sing|Person[psor]=2|Polite=Form", "12"),
]
CASES_PLAIN = [
    (["ның", "нің", "дың", "дің", "тың", "тің"], "Case=Gen"),
    (["ға", "ге", "қа", "ке"], "Case=Dat"),
    (["ны", "ні", "ды", "ді", "ты", "ті"], "Case=Acc"),
    (["да", "де", "та", "те", "дағы", "дегі", "тағы", "тегі"], "Case=Loc"),
    (["дан", "ден", "тан", "тен", "нан", "нен"], "Case=Abl"),
    (["мен", "бен", "пен", "менен", "бенен", "пенен"], "Case=Ins"),
]
CASES_AFTER_POSS3 = [          # после притяжательного аффикса 3 л. падежи присоединяются через -н-
    (["ның", "нің"], "Case=Gen"),
    (["на", "не"], "Case=Dat"),
    (["н"], "Case=Acc"),
    (["нда", "нде", "ндағы", "ндегі"], "Case=Loc"),
    (["нан", "нен"], "Case=Abl"),
    (["мен", "менен"], "Case=Ins"),
]
CASES_AFTER_POSS12 = [
    (["ның", "нің", "дың", "дің"], "Case=Gen"),
    (["а", "е", "ға", "ге"], "Case=Dat"),
    (["ды", "ді", "ны", "ні"], "Case=Acc"),
    (["да", "де", "нда", "нде"], "Case=Loc"),
    (["дан", "ден", "нан", "нен"], "Case=Abl"),
    (["мен", "бен", "пен"], "Case=Ins"),
]


def _nominal_rules():
    """Все цепочки «мн. ч.? + притяж.? + падеж?» → (суффикс, признаки)."""
    rules = {}

    def put(suffix, feats, name):
        if suffix:
            rules.setdefault(suffix, []).append((join_feats(feats), name))

    plural_opts = [("", "", "")] + [(a, PLURAL[1], "PL") for a in PLURAL[0]]
    for pl, pl_f, pl_n in plural_opts:
        # без притяжательности
        for c_affixes, c_f in CASES_PLAIN:
            for c in c_affixes:
                put(pl + c, join_feats(pl_f, c_f), f"{pl_n}+{c_f}")
        put(pl, join_feats(pl_f, "Case=Nom"), f"{pl_n}+Nom")
        # с притяжательностью
        for p_affixes, p_f, kind in POSSESSIVES:
            if pl and kind == "3":
                p_affixes = ["ы", "і"]            # лар+ы, лер+і
            cases = CASES_AFTER_POSS3 if kind == "3" else CASES_AFTER_POSS12
            for p in p_affixes:
                put(pl + p, join_feats(pl_f, p_f, "Case=Nom"), f"{pl_n}+POSS+Nom")
                for c_affixes, c_f in cases:
                    for c in c_affixes:
                        put(pl + p + c, join_feats(pl_f, p_f, c_f), f"{pl_n}+POSS+{c_f}")
    return rules


NOMINAL_RULES = _nominal_rules()

# ============================================================ глаголы
# (алломорфы окончания, UPOS, признаки). Порядок не важен: выигрывает самый длинный аффикс.
_PASS = ["ыл", "іл", "л", "ын", "ін", "н"]
VERBAL_ENDINGS = [
    (["ды", "ді", "ты", "ті"], "Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin"),
    (["дым", "дім", "тым", "тім"], "Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin"),
    (["дық", "дік", "тық", "тік"], "Mood=Ind|Number=Plur|Person=1|Tense=Past|VerbForm=Fin"),
    (["дыңыз", "діңіз", "тыңыз", "тіңіз"], "Mood=Ind|Number=Sing|Person=2|Polite=Form|Tense=Past|VerbForm=Fin"),
    (["ады", "еді", "йды", "йді"], "Aspect=Hab|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"),
    (["амын", "емін", "ймын", "ймін", "ам", "ем", "ймн"], "Aspect=Hab|Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin"),
    (["амыз", "еміз", "ймыз", "йміз"], "Aspect=Hab|Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin"),
    (["асыз", "есіз", "йсыз", "йсіз"], "Aspect=Hab|Mood=Ind|Number=Sing|Person=2|Polite=Form|Tense=Pres|VerbForm=Fin"),
    (["майды", "мейді", "байды", "бейді", "пайды", "пейді"],
     "Aspect=Hab|Mood=Ind|Number=Sing|Person=3|Polarity=Neg|Tense=Pres|VerbForm=Fin"),
    (["мады", "меді", "бады", "беді", "пады", "педі"],
     "Mood=Ind|Number=Sing|Person=3|Polarity=Neg|Tense=Past|VerbForm=Fin"),
    (["ыпты", "іпті", "пты", "пті"], "Evident=Fh|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin"),
    (["ған", "ген", "қан", "кен"], "Tense=Past|VerbForm=Part"),
    (["маған", "меген", "баған", "беген", "паған", "пеген"], "Polarity=Neg|Tense=Past|VerbForm=Part"),
    (["атын", "етін", "йтын", "йтін"], "Aspect=Imp|VerbForm=Part"),
    (["ғанда", "генде", "қанда", "кенде"], "Case=Loc|Tense=Past|VerbForm=Ger"),
    (["ғаннан", "геннен", "қаннан", "кеннен"], "Case=Abl|Tense=Past|VerbForm=Ger"),
    (["ғанын", "генін", "қанын", "кенін"], "Case=Acc|Number[psor]=Plur,Sing|Person[psor]=3|Tense=Past|VerbForm=Ger"),
    (["ып", "іп"], "Aspect=Perf|VerbForm=Conv"),
    (["п"], "Aspect=Perf|VerbForm=Conv"),
    (["а", "е", "й"], "Aspect=Imp|VerbForm=Inf"),
    (["у"], "Case=Nom|VerbForm=Ger"),
    (["уға", "уге"], "Case=Dat|VerbForm=Ger"),
    (["уды", "уді"], "Case=Acc|VerbForm=Ger"),
    (["уда", "уде"], "Case=Loc|VerbForm=Ger"),
    (["удан", "уден"], "Case=Abl|VerbForm=Ger"),
    (["удың", "удің"], "Case=Gen|VerbForm=Ger"),
    (["уы", "уі"], "Case=Nom|Number[psor]=Plur,Sing|Person[psor]=3|VerbForm=Ger"),
    (["уын", "уін"], "Case=Acc|Number[psor]=Plur,Sing|Person[psor]=3|VerbForm=Ger"),
    (["уына", "уіне"], "Case=Dat|Number[psor]=Plur,Sing|Person[psor]=3|VerbForm=Ger"),
    (["са", "се"], "Mood=Cnd|Number=Sing|Person=3|VerbForm=Conv"),
    (["сам", "сем"], "Mood=Cnd|Number=Sing|Person=1|VerbForm=Conv"),
    (["сақ", "сек"], "Mood=Cnd|Number=Plur|Person=1|VerbForm=Conv"),
    (["сын", "сін"], "Mood=Opt|Number=Sing|Person=3|VerbForm=Fin"),
    (["айық", "ейік", "йық", "йік"], "Mood=Opt|Number=Plur|Person=1|VerbForm=Fin"),
    (["ыңыз", "іңіз", "ңыз", "ңіз"], "Mood=Imp|Number=Sing|Person=2|Polite=Form|VerbForm=Fin"),
    (["ар", "ер", "р"], "Tense=Fut|VerbForm=Part"),
    (["мақ", "мек", "бақ", "бек", "пақ", "пек"], "Mood=Des|VerbForm=Part"),
]


def _verbal_rules():
    rules = {}
    for affixes, feats in VERBAL_ENDINGS:
        for a in affixes:
            rules.setdefault(a, []).append((feats, "VERB"))
            # страдательный залог: основа + -ыл/-іл/-л/-н + окончание (только для окончаний с гласной/у)
            if a[0] in VOWELS or a[0] in "дтгғқк":
                for p in _PASS:
                    rules.setdefault(p + a, []).append((join_feats(feats, "Voice=Pass"), "VERB+PASS"))
    return rules


VERBAL_RULES = _verbal_rules()

# ====================================================== словообразование
DERIVATIONAL = [
    (["лы", "лі", "ды", "ді", "ты", "ті"], "ADJ", "_", "ADJ:-лы"),
    (["сыз", "сіз"], "ADJ", "_", "ADJ:-сыз"),
    (["шыл", "шіл"], "ADJ", "_", "ADJ:-шыл"),
    (["ғы", "гі", "қы", "кі"], "ADJ", "_", "ADJ:-ғы"),
    (["дай", "дей", "тай", "тей"], "ADJ", "_", "ADJ:-дай"),
    (["лық", "лік", "дық", "дік", "тық", "тік"], "NOUN", "Case=Nom", "NOUN:-лық"),
    (["шы", "ші"], "NOUN", "Case=Nom", "NOUN:-шы"),
    (["ша", "ше"], "ADV", "_", "ADV:-ша"),
    (["ятын", "итін"], "VERB", "Aspect=Imp|VerbForm=Part", "VERB:-итін"),
]


def _has_vowel(s):
    return any(ch in VOWELS for ch in s)


class MorphAnalyzer:
    """Кандидаты-разборы слова. train_lexicon — словарь словоформ из train (может быть пустым)."""

    MIN_STEM = 2

    def __init__(self, train_sentences=None):
        self.closed = closed_class_lexicon()
        self.train = defaultdict(Counter)       # слово → Counter((UPOS, FEATS, lemma))
        self.stems = defaultdict(Counter)       # основа (лемма) → Counter(UPOS): «бар» → VERB
        for s in train_sentences or []:
            for t, l, u, f in zip(s["tokens"], s["lemmas"], s["upos"], s["feats"]):
                self.train[t.lower()][(u, f, l)] += 1
                self.stems[l.lower()][u] += 1

    # ---------------------------------------------------------- кандидаты
    def analyze(self, token, sentence_initial=False):
        low = token.lower()
        out = []
        if PUNCT_RE.match(token):
            return [Analysis("PUNCT", "_", token, "punct", 10.0, "punct")]
        if NUM_ORD_RE.match(low):
            out.append(Analysis("NUM", "NumType=Ord", token, "number", 9.0, "digits+ord"))
        elif NUM_RE.match(low):
            return [Analysis("NUM", "NumType=Card", token, "number", 9.0, "digits")]
        elif NUM_SUFFIX_RE.match(low):
            out.append(Analysis("NUM", "NumType=Card", token, "number", 8.0, "digits+suffix"))

        # 2. словарь из train: относительная частота → оценка 6..7
        if low in self.train:
            c = self.train[low]
            total = sum(c.values())
            for (u, f, lem), n in c.most_common():
                out.append(Analysis(u, f, lem, "train_lexicon", 6.0 + n / total, "train"))
        # 3. служебные слова, составленные вручную
        for u, f in self.closed.get(low, []):
            out.append(Analysis(u, f, low, "closed_class", 5.0, "closed"))
        # 4. морфотактика
        out += self._suffix_candidates(token, low, sentence_initial)
        # 5. умолчание
        is_proper = token[:1].isupper() and not sentence_initial and not token.isupper()
        out.append(Analysis("PROPN" if is_proper else "NOUN", "Case=Nom", token, "default",
                            2.6 if is_proper else 1.4, "default"))
        if token[:1].isupper() and sentence_initial:
            out.append(Analysis("PROPN", "Case=Nom", token, "default", 1.3, "default"))
        return self._dedup(out)

    def _suffix_candidates(self, token, low, sentence_initial):
        out = []
        proper = token[:1].isupper() and not sentence_initial
        nominal_upos = "PROPN" if proper else "NOUN"
        for k in range(1, min(len(low) - self.MIN_STEM, 14) + 1):
            suf, stem = low[-k:], low[:-k]
            if not _has_vowel(stem):
                continue
            bonus = 0.25 * k                    # длинный аффикс — более специфичное правило
            known = self.stems.get(stem)        # основа известна из словаря лемм?
            verb_stem = bool(known) and known.most_common(1)[0][0] in ("VERB", "AUX")
            noun_stem = bool(known) and known.most_common(1)[0][0] in ("NOUN", "PROPN", "ADJ", "NUM", "PRON")
            for feats, name in NOMINAL_RULES.get(suf, []):
                out.append(Analysis(nominal_upos, feats, stem, "suffix",
                                    1.5 + bonus + (1.0 if proper else 0) + (1.5 if noun_stem else 0)
                                    - (1.5 if verb_stem else 0), f"N:{name}:-{suf}"))
            if not proper:
                for feats, name in VERBAL_RULES.get(suf, []):
                    # односимвольные глагольные аффиксы очень неоднозначны — штраф
                    penalty = 1.0 if k == 1 else 0.0
                    upos = "AUX" if stem == "е" else "VERB"   # «бол», «кел» и т.п. чаще самостоятельные
                    out.append(Analysis(upos, feats, stem, "suffix",
                                        1.4 + bonus - penalty + (1.5 if verb_stem else 0)
                                        - (1.0 if noun_stem else 0), f"V:{name}:-{suf}"))
            for affixes, upos, feats, name in DERIVATIONAL:
                if suf in affixes and not proper:
                    out.append(Analysis(upos, feats, low, "suffix", 1.2 + bonus, name))
        return out

    @staticmethod
    def _dedup(cands):
        """Один кандидат на пару (UPOS, FEATS) — с наибольшей оценкой."""
        best = {}
        for a in cands:
            key = (a.upos, a.feats)
            if key not in best or a.score > best[key].score:
                best[key] = a
        return sorted(best.values(), key=lambda a: -a.score)

    # ---------------------------------------------- основа для NER-газеттира
    def nominal_stems(self, token):
        """Все возможные основы после отсечения именных аффиксов, от длинной к короткой:
        «Қазақстанның» → «Қазақстанны», «Қазақстан», «Қазақс», …  Какая верна — решает
        тот, кто вызывает (например, газеттир: основа должна быть известной сущностью)."""
        low = token.lower()
        return [token[:-k] for k in range(1, min(len(low) - 2, 10) + 1)
                if low[-k:] in NOMINAL_RULES and _has_vowel(low[:-k])]

    def nominal_stem(self, token):
        """Самая короткая основа (максимальная отсечка аффиксов) — грубая эвристика."""
        stems = self.nominal_stems(token)
        return stems[-1] if stems else token


class RuleTagger:
    """Контекстные правила поверх кандидатов анализатора и выбор лучшего разбора."""

    FINAL_PUNCT = {".", "!", "?", "…", "..."}

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def candidates(self, tokens):
        """Кандидаты для каждого токена предложения, уже с учётом контекста."""
        cands = [self.analyzer.analyze(t, sentence_initial=(i == 0)) for i, t in enumerate(tokens)]
        last_word = max((i for i, t in enumerate(tokens) if not PUNCT_RE.match(t)), default=-1)
        out = []
        for i, cs in enumerate(cands):
            adjusted = []
            for a in cs:
                s = a.score
                # К1. последнее слово предложения обычно сказуемое: финитные формы вперёд
                if i == last_word and a.source in ("suffix", "default"):
                    if "VerbForm=Fin" in a.feats:
                        s += 1.5
                    elif a.upos in ("NOUN", "PROPN") and "Case=Acc" in a.feats:
                        s -= 1.0
                # К2. не последнее слово: финитные глагольные догадки маловероятны
                if i != last_word and a.source == "suffix" and "VerbForm=Fin" in a.feats:
                    s -= 0.8
                # К3. «мен» в начале предложения — местоимение «я», иначе союз
                if tokens[i].lower() == "мен" and a.source == "closed_class":
                    s += 0.5 if (i == 0) == (a.upos == "PRON") else -0.5
                # К4. перед вспомогательным глаголом стоит деепричастие на -ып/-а
                nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
                if nxt in ("жатыр", "отыр", "тұр", "жүр", "еді", "кетті", "қалды", "алды", "берді") \
                        and a.source == "suffix" and "VerbForm=Conv" in a.feats:
                    s += 1.0
                adjusted.append(Analysis(a.upos, a.feats, a.lemma, a.source, s, a.rule))
            out.append(sorted(adjusted, key=lambda a: -a.score))
        return out

    def tag(self, tokens):
        return [c[0] for c in self.candidates(tokens)]
