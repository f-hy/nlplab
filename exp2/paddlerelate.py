# -*- coding : utf-8 -*-
# @Time      : 2023/3/7 22:16
# @Author    : Friendy  付
# @File      : paddlerelate
# @Software  : PyCharm
import paddle
import jieba
f = open(file="Chinese.txt", mode="r", encoding="utf8")
ct = f.read()
f.close()
jieba.enable_paddle()
seg_list = jieba.cut(ct, use_paddle=True)  #paddle模式
print("Paddle Mode: " + '/'.join(list(seg_list)))
seg_list = jieba.cut(ct, cut_all=True)  # 全模式
print("Full Mode: " + "/ ".join(seg_list))
seg_list = jieba.cut(ct, cut_all=False)  # 精确模式
print("Default Mode: " + "/ ".join(seg_list))
seg_list = jieba.cut_for_search(ct)  # 搜索引擎模式
print("Search Mode: " + "/ ".join(seg_list))
print("\n\n")
jieba.load_userdict("./Userdict.txt")
# jieba.enable_paddle()
seg_list = jieba.cut(ct, use_paddle=True)  #paddle模式
print("Paddle Mode: " + '/'.join(list(seg_list)))
seg_list = jieba.cut(ct, cut_all=True)  # 全模式
print("Full Mode: " + "/ ".join(seg_list))
seg_list = jieba.cut(ct, cut_all=False)  # 精确模式
print("Default Mode: " + "/ ".join(seg_list))
seg_list = jieba.cut_for_search(ct)  # 搜索引擎模式
print("Search Mode: " + "/ ".join(seg_list))