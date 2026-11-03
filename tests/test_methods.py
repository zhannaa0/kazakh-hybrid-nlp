"""Тесты методов: rule-based морфология и NER, гибрид. Нейросеть — test_neural.py."""

import math

import pytest

from src.hybrid.combine import combine_token, hybrid_ner_sentence, hybrid_sequence_tagging
from src.rules.lexicon import closed_class_lexicon
from src.rules.morphology import MorphAnalyzer, RuleTagger, join_feats
from src.rules.ner import RuleNER


# ------------------------------------------------------------ морфология

@pytest.fixture(scope="module")
def analyzer():
    return MorphAnalyzer()          # только ручные правила, без словаря из train


def best(analyzer, word, initial=False):
    return analyzer.analyze(word, sentence_initial=initial)[0]


def test_join_feats_ud_order():
    assert join_feats("Number=Plur", "Case=Loc", "Number[psor]=Plur,Sing|Person[psor]=3") == \
        "Case=Loc|Number=Plur|Number[psor]=Plur,Sing|Person[psor]=3"


@pytest.mark.parametrize("word, upos, feats", [
    ("кітаптарда", "NOUN", "Case=Loc|Number=Plur"),
    ("үйіміздің", "NOUN", "Case=Gen|Number[psor]=Plur|Person[psor]=1"),
    ("баласына", "NOUN", "Case=Dat|Number[psor]=Plur,Sing|Person[psor]=3"),
    ("бізге", "PRON", "Case=Dat|Number=Plur|Person=1|PronType=Prs"),
    ("және", "CCONJ", "_"),
    ("2014", "NUM", "NumType=Card"),
    ("1-інші", "NUM", "NumType=Ord"),
    (",", "PUNCT", "_"),
])
def test_analyzer_best_candidate(analyzer, word, upos, feats):
    a = best(analyzer, word)
    assert (a.upos, a.feats) == (upos, feats)


def test_verbal_candidates_present(analyzer):
    feats = {a.feats for a in analyzer.analyze("жазылған")}
    assert "Tense=Past|VerbForm=Part|Voice=Pass" in feats
    feats = {a.feats for a in analyzer.analyze("бармайды")}
    assert "Aspect=Hab|Mood=Ind|Number=Sing|Person=3|Polarity=Neg|Tense=Pres|VerbForm=Fin" in feats


def test_proper_noun_by_capital_letter(analyzer):
    assert best(analyzer, "Астанада").upos == "PROPN"
    assert best(analyzer, "Астанада").feats == "Case=Loc"


def test_context_rule_sentence_final_verb(analyzer):
    tagger = RuleTagger(analyzer)
    tags = tagger.tag(["Ол", "үйге", "барды", "."])
    assert tags[2].upos == "VERB" and "Tense=Past" in tags[2].feats


def test_train_lexicon_and_stem_lexicon():
    train = [{"tokens": ["бар"], "lemmas": ["бар"], "upos": ["VERB"], "feats": ["Mood=Imp"]}]
    a = MorphAnalyzer(train)
    assert best(a, "бар").upos == "VERB"                          # словарь словоформ
    assert best(a, "барып").upos == "VERB"                        # основа известна как глагольная


def test_nominal_stem():
    a = MorphAnalyzer()
    assert a.nominal_stem("Қазақстанның") == "Қазақстан"
    assert a.nominal_stem("Алматыда") == "Алматы"


def test_closed_lexicon_has_paradigms():
    lex = closed_class_lexicon()
    assert ("PRON", "Case=Abl|Number=Sing|Person=3|PronType=Prs") in lex["одан"]
    assert ("NUM", "NumType=Ord") in lex["бірінші"]


# --------------------------------------------------------------------- NER

def _ner_train():
    return [{"tokens": ["Қазақстан", "дамып", "келеді", "."], "ner": ["B-GPE", "O", "O", "O"]},
            {"tokens": ["Қазақстанның", "астанасы", "."], "ner": ["B-GPE", "O", "O"]},
            {"tokens": ["Ауыл", "кеші", "."], "ner": ["B-ORGANISATION", "O", "O"]},
            {"tokens": ["Ауыл", "үлкен", "."], "ner": ["O", "O", "O"]},
            {"tokens": ["Ауыл", "әдемі", "."], "ner": ["O", "O", "O"]}]


def test_ner_patterns():
    r = RuleNER()
    assert r.tag(["бес", "жүз", "мың", "теңге", "бөлінді"]) == ["B-MONEY", "I-MONEY", "I-MONEY", "I-MONEY", "O"]
    assert r.tag(["30", "%-ке", "өсті"]) == ["B-PERCENTAGE", "I-PERCENTAGE", "O"]
    assert r.tag(["2020", "жылы", "басталды"]) == ["B-DATE", "I-DATE", "O"]
    assert r.tag(["20", "тонна", "астық"]) == ["B-QUANTITY", "I-QUANTITY", "O"]
    assert r.tag(["екіншіден", ","])[0] == "B-ORDINAL"
    assert r.tag(["бірге", "барды"]) == ["O", "O"]


def test_ner_gazetteer_with_stem_and_reliability():
    r = RuleNER(_ner_train(), min_rate=0.5, min_count=2)
    assert r.tag(["Қазақстанда", "қар", "жауды"])[0] == "B-GPE"      # основа «Қазақстан»
    assert r.tag(["Ауыл", "жақсы"])[0] == "O"                        # сущность лишь в 1 из 3 случаев


# ------------------------------------------------------------------- гибрид

def test_hard_constraint_for_punct():
    y, how = combine_token("NOUN", [["NOUN", 0.9]], [["PUNCT", "_", 10.0, "punct"]], "upos", lam=0.5)
    assert (y, how) == ("PUNCT", "hard")


def test_lambda_zero_is_pure_neural_and_large_lambda_follows_rules():
    probs = [["NOUN", 0.55], ["VERB", 0.45]]
    rules = [["VERB", "_", 4.0, "suffix"], ["NOUN", "_", 1.0, "default"]]
    assert combine_token("NOUN", probs, rules, "upos", lam=0)[0] == "NOUN"
    assert combine_token("NOUN", probs, rules, "upos", lam=2.0)[0] == "VERB"


def test_split_feats_scoring():
    probs = {"Case": [["Nom", 0.4], ["Loc", 0.6]], "Number": [["_", 0.9], ["Plur", 0.1]]}
    rules = [["NOUN", "Case=Nom", 5.0, "train_lexicon"]]
    y, _ = combine_token("Case=Loc", probs, rules, "feats", lam=1.0, mode="split")
    assert y == "Case=Nom"


def test_hybrid_sequence_requires_same_sentences():
    n = [{"id": 1, "tokens": ["а"], "gold": ["X"], "pred": ["X"], "probs": [[["X", 1.0]]]}]
    r = [{"id": 1, "tokens": ["б"], "candidates": [[]]}]
    with pytest.raises(ValueError):
        hybrid_sequence_tagging(n, r, "upos", 1.0)


def test_hybrid_ner_pattern_override_and_gazetteer_fill():
    neural = ["B-CARDINAL", "I-CARDINAL", "O", "O", "O"]
    probs = [[["B-CARDINAL", 0.5]], [["I-CARDINAL", 0.5]], [["O", 0.6]], [["O", 0.99]], [["O", 0.99]]]
    spans = [["MONEY", 0, 3, 0.95, "pattern"], ["GPE", 4, 5, 0.97, "gazetteer"]]
    out = hybrid_ner_sentence(neural, probs, spans, tau_p=0.9, gamma=0.8, tau_g=0.9)
    assert out == ["B-MONEY", "I-MONEY", "I-MONEY", "O", "B-GPE"]
    # высокие пороги — гибрид совпадает с нейросетью
    assert hybrid_ner_sentence(neural, probs, spans, 2.0, 0.8, 2.0) == neural
    assert math.isclose(1.0, 1.0)
