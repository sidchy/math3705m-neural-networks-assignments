from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ExcelSource:
    code: str
    filename: str
    kind: str
    read_kwargs: dict
    rename: dict[str | int, str]
    text_columns: Sequence[str]
    key_columns: Sequence[str]


SOURCES = [
    ExcelSource("01", "01_【20260327最终版】温州方言词典.xlsx", "lexicon", {}, {
        "词条简体": "entry", "释义简体": "definition", "方言例句": "dialect", "普通话翻译": "translation"
    }, ("entry", "definition", "dialect", "translation"), ("entry", "definition")),
    ExcelSource("02", "02_副本温州话词语考释1-4199(1)(1).xlsx", "lexicon", {"header": None}, {
        0: "entry", 1: "definition", 2: "dialect", 3: "translation"
    }, ("entry", "definition", "dialect", "translation"), ("entry", "definition")),
    ExcelSource("03", "03_温州话词语考释8595-12651.xlsx", "lexicon", {"header": None}, {
        0: "entry", 1: "definition", 2: "dialect", 3: "translation"
    }, ("entry", "definition", "dialect", "translation"), ("entry", "definition")),
    ExcelSource("04", "04_【终版】活色生香温州话.xlsx", "lexicon", {}, {
        "类型": "category", "原文": "entry", "普通话释义": "definition"
    }, ("category", "entry", "definition"), ("entry", "definition")),
    ExcelSource("05", "05_温州话资源库-词汇-323交付版.xlsx", "lexicon", {}, {
        "词条": "entry", "释义": "definition"
    }, ("entry", "definition"), ("entry", "definition")),
    ExcelSource("06", "06_天空对话_上传版.xlsx", "parallel", {}, {
        "温州话": "dialect", "普通话": "translation"
    }, ("dialect", "translation"), ("dialect", "translation")),
    ExcelSource("07", "07_温州方言论文语料.（3.24）xlsx.xlsx", "parallel", {}, {
        "dialect": "dialect", "translation": "translation", "source_file": "source_doc"
    }, ("dialect", "translation", "source_doc"), ("dialect", "translation")),
    ExcelSource("08", "08_【3.25（2）】林老师.xlsx", "parallel", {}, {
        "clean_text": "dialect", "mandarin_translation": "translation"
    }, ("dialect", "translation"), ("dialect", "translation")),
    ExcelSource("09", "09_4_10上传_大蒙讲温州_清洗后v4.xlsx", "parallel", {}, {
        "wenzhou": "dialect", "mandarin": "translation"
    }, ("dialect", "translation"), ("dialect", "translation")),
    ExcelSource("12", "新事物名词.xlsx", "wordlist", {"header": None}, {
        0: "entry"
    }, ("entry",), ("entry",)),
    ExcelSource("13", "温州地名300个.xlsx", "wordlist", {"header": None}, {
        0: "entry"
    }, ("entry",), ("entry",)),
]
