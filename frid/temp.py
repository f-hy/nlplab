# -*- coding : utf-8 -*-
# @Time      : 2023/3/14 10:17
# @Author    : Friendy  付
# @File      : temp
# @Software  : PyCharm
"""
69t112: jieba,nltk,SnowNLP,THULAC,NLPIR(PyNLPIR),spacy
pyhanlp: StandfordCoreNLP,HanLP
"""


def get_txt(file_path: str):
    f = open(file=file_path, mode="r", encoding="utf8")
    txt = f.read()
    f.close()
    return txt