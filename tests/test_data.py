"""Тесты модуля данных на маленьких файлах-образцах (настоящие датасеты не нужны)."""

import json

import pytest

from src.data.clean import clean_kaznerd, fix_homoglyphs, repair_iob2
from src.data.prepare import prepare
from src.data.readers import read_conllu, read_iob2
from src.data.splits import stratified_split

IOB2_SAMPLE = """Астана B-GPE
қаласы I-GPE
. O

Aстана B-GPE
I-PERSON O
. O
"""

CONLLU_SAMPLE = """# sent_id = story.tagged.txt:1:0
# text = Баласыз үй.
1-2	Баласыз	_	_	_	_	_	_	_	_
1	бала	бала	NOUN	n	_	3	nmod	_	_
2	сыз	сыз	ADP	post	_	1	case	_	_
3	үй	үй	NOUN	n	Case=Nom	0	root	_	SpaceAfter=No
4	.	.	PUNCT	sent	_	3	punct	_	_

"""


# --------------------------------------------------------------- чтение

def test_read_iob2(tmp_path):
    p = tmp_path / "IOB2_train.txt"
    p.write_text(IOB2_SAMPLE, encoding="utf-8")
    sents = read_iob2(p, id_prefix="kaznerd-train")
    assert len(sents) == 2
    assert sents[0]["tokens"] == ["Астана", "қаласы", "."]
    assert sents[0]["ner"] == ["B-GPE", "I-GPE", "O"]
    assert sents[0]["id"] == "kaznerd-train-000001"


def test_read_iob2_rejects_bad_line(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("Астана B-GPE лишнее\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.txt:1"):
        read_iob2(p)


def test_read_conllu_skips_multiword_ranges(tmp_path):
    p = tmp_path / "x.conllu"
    p.write_text(CONLLU_SAMPLE, encoding="utf-8")
    sents, stats = read_conllu(p)
    assert stats["multiword_tokens"] == 1
    s = sents[0]
    assert s["tokens"] == ["бала", "сыз", "үй", "."]          # синтаксические слова
    assert s["upos"] == ["NOUN", "ADP", "NOUN", "PUNCT"]
    assert s["feats"][2] == "Case=Nom"
    assert s["doc"] == "story"


# --------------------------------------------------------------- очистка

@pytest.mark.parametrize("before, after", [
    ("Aстана", "Астана"),                 # латинская A
    ("мемлекеттiк", "мемлекеттік"),       # латинская i → казахская і
    ("IT-кабинет", "IT-кабинет"),         # части через дефис — по отдельности, IT не трогаем
    ("COVID-19-бен", "COVID-19-бен"),
    ("Mall-дың", "Mall-дың"),
    ("Astana", "Astana"),                 # полностью латинское слово не трогаем
])
def test_fix_homoglyphs(before, after):
    assert fix_homoglyphs(before) == after


def test_repair_iob2():
    assert repair_iob2(["O", "I-PER", "I-PER"]) == ["O", "B-PER", "I-PER"]
    assert repair_iob2(["B-PER", "I-LOC"]) == ["B-PER", "B-LOC"]
    assert repair_iob2(["B-PER", "I-PER", "O"]) == ["B-PER", "I-PER", "O"]


def test_clean_logs_every_change():
    s = [{"id": "x", "tokens": ["Aстана", "қаласы"], "ner": ["I-GPE", "I-GPE"]}]
    log = []
    clean_kaznerd(s, "train", log)
    assert s[0]["tokens"][0] == "Астана" and s[0]["ner"] == ["B-GPE", "I-GPE"]
    assert {r["rule"] for r in log} == {"homoglyph", "iob2_repair"}


# --------------------------------------------------------------- сплиты

def _ud_pool():
    return [{"id": f"{d}:{i}", "doc": d} for d, n in (("big", 50), ("mid", 20), ("tiny", 3))
            for i in range(n)]


def test_stratified_split_is_deterministic_and_complete():
    pool = _ud_pool()
    a, b = stratified_split(pool, seed=42), stratified_split(pool, seed=42)
    assert a == b
    all_ids = a["train"] + a["dev"] + a["test"]
    assert sorted(all_ids) == sorted(s["id"] for s in pool)      # ничего не потеряно, без повторов
    assert len(set(all_ids)) == len(all_ids)


def test_stratified_split_keeps_proportions_per_doc():
    split = stratified_split(_ud_pool(), seed=1)
    count = lambda sp, d: sum(i.startswith(d + ":") for i in split[sp])
    assert count("dev", "big") == 5 and count("test", "big") == 5
    assert count("dev", "mid") == 2 and count("test", "mid") == 2
    assert count("train", "tiny") == 3                              # маленький текст целиком в train


# --------------------------------------------------------- весь конвейер

def test_prepare_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    (raw / "kaznerd").mkdir(parents=True)
    (raw / "ud-ktb").mkdir()
    for fn in ("IOB2_train.txt", "IOB2_valid.txt", "IOB2_test.txt"):
        (raw / "kaznerd" / fn).write_text(IOB2_SAMPLE, encoding="utf-8")
    for fn in ("kk_ktb-ud-train.conllu", "kk_ktb-ud-test.conllu"):
        (raw / "ud-ktb" / fn).write_text(CONLLU_SAMPLE * 3, encoding="utf-8")

    m = prepare(raw, tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "kaznerd" / "dev.jsonl").exists()
    assert (out / "udktb" / "official" / "test.jsonl").exists()
    assert len(m["sources"]["kaznerd"]["files"]["IOB2_train.txt"]) == 64     # sha256
    assert m["cleaning"]["homoglyph"] == 3 and "iob2_repair" not in m["cleaning"]
    first = json.loads((out / "kaznerd" / "train.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert first["tokens"][0] == "Астана"                                      # очищено
