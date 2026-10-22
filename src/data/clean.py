"""Первичная очистка. Каждое изменение записывается в журнал (log), чтобы
в диссертации можно было точно сказать, что и сколько раз исправлено.

Правила (применяются одинаково ко всем сплитам и ко всем методам):

  1. unicode_nfc  — Unicode-нормализация NFC (одна и та же буква всегда
                    записана одной кодовой точкой).
  2. homoglyph    — латинская буква-двойник внутри кириллического слова
                    заменяется кириллической: «Aстана» (лат. A) → «Астана»,
                    «мемлекеттiк» (лат. i) → «мемлекеттік». Слова вроде
                    «IT-кабинет» не трогаем: части через дефис проверяются отдельно.
  3. iob2_repair  — только KazNERD: «I-X» без предшествующего «B-X»/«I-X»
                    превращается в «B-X» (стандартное исправление IOB2).

Предложения мы НЕ удаляем: дубликаты и утечки только фиксируются
(см. quality.py), чтобы официальные сплиты остались сравнимы со статьями.
"""

import re
import unicodedata

# Латинские буквы, визуально совпадающие с кириллическими (в т.ч. казахскими)
HOMOGLYPHS = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "I": "І", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "Ү",
    "a": "а", "c": "с", "e": "е", "h": "һ", "i": "і", "o": "о", "p": "р", "x": "х", "y": "у",
}
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
SEGMENT_SPLIT = re.compile(r"([-–—/])")    # скобки в шаблоне сохраняют разделители


def fix_homoglyphs(token):
    """Вернуть токен с исправленными двойниками (или тот же токен)."""
    parts = SEGMENT_SPLIT.split(token)
    for k, seg in enumerate(parts):
        if CYRILLIC.search(seg) and LATIN.search(seg):
            latin_letters = set(LATIN.findall(seg))
            if latin_letters <= HOMOGLYPHS.keys():     # исправляем, только если все латинские буквы — двойники
                parts[k] = "".join(HOMOGLYPHS.get(ch, ch) for ch in seg)
    return "".join(parts)


def repair_iob2(tags):
    """I-X без начала сущности → B-X. Возвращает новый список тегов."""
    fixed, prev = [], "O"
    for tag in tags:
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            tag = "B-" + tag[2:]
        fixed.append(tag)
        prev = tag
    return fixed


def _clean_field(sentence, field, rules, log, split):
    values = sentence[field]
    for i, value in enumerate(values):
        for rule, fn in rules:
            new = fn(value)
            if new != value:
                log.append({"split": split, "id": sentence["id"], "field": field,
                            "position": i, "rule": rule, "before": value, "after": new})
                value = new
        values[i] = value


TEXT_RULES = [
    ("unicode_nfc", lambda s: unicodedata.normalize("NFC", s)),
    ("homoglyph", fix_homoglyphs),
]


def clean_kaznerd(sentences, split, log):
    """Очистить предложения KazNERD на месте; изменения дописываются в log."""
    for s in sentences:
        _clean_field(s, "tokens", TEXT_RULES, log, split)
        repaired = repair_iob2(s["ner"])
        for i, (old, new) in enumerate(zip(s["ner"], repaired)):
            if old != new:
                log.append({"split": split, "id": s["id"], "field": "ner", "position": i,
                            "rule": "iob2_repair", "before": old, "after": new})
        s["ner"] = repaired
    return sentences


def clean_ud(sentences, split, log):
    """Очистить предложения UD-KTB на месте (токены и леммы)."""
    for s in sentences:
        _clean_field(s, "tokens", TEXT_RULES, log, split)
        _clean_field(s, "lemmas", TEXT_RULES, log, split)
        s["text"] = fix_homoglyphs(unicodedata.normalize("NFC", s["text"]))
    return sentences
