# -*- coding : utf-8 -*-
# @Time      : 2023/2/28 10:03
# @Author    : Friendy  付
# @File      : lab1
# @Software  : PyCharm
from nltk.tokenize import word_tokenize
from nltk import bigrams, FreqDist
from math import log


class Gram:
    def __init__(self, train_file_path: str, test_file_path: str) -> None:
        f = open(file=train_file_path, mode="r+", encoding="utf8")
        self.trainset = f.read()
        f.close()
        f = open(file=test_file_path, mode="r+", encoding="utf8")
        self.testset = f.read()
        f.close()
        self.preprocessing()
        super().__init__()

    def init(self, is_bio: bool):
        self.is_bio = is_bio
        self.gramsDist = FreqDist()  # trainset+testset的 -gram词频数字典
        if self.is_bio:
            self.w2gram = {}  # 可能存在的以w为开头的2-gram的种类数量

    def preprocessing(self):  # 读取数据 小写 替换符号 分句
        symbols = ',.?!:;<>/'
        # self.doc.translate(dict.fromkeys(map(ord,string.punctuation)))
        self.trainset = self.trainset.lower().translate(str.maketrans(symbols, ' ' * len(symbols))).split("__eou__")
        self.testset = self.testset.lower().translate(str.maketrans(symbols, ' ' * len(symbols))).split("__eou__")

    def train(self):
        for sentence in self.trainset:
            # 获取该句各个词的词频
            if self.is_bio:
                sWordFreq = FreqDist(bigrams(word_tokenize("BOS "+sentence+" EOS")))
            else:
                sWordFreq = FreqDist(word_tokenize(sentence))  # 每一句的词频数字典
            for j in sWordFreq:
                if j in self.gramsDist:
                    self.gramsDist[j] += sWordFreq[j]
                else:
                    self.gramsDist[j] = sWordFreq[j]
                    if self.is_bio:
                        if j[0] in self.w2gram:
                            self.w2gram[j[0]] += 1
                        else:
                            self.w2gram[j[0]] = 1

    def add_OOV(self):  # 加入未登录词 Out-of-vocabulary
        # 由于将每种未出现的2-gram一一列举会生成vacab size * vocab size大小的bigramsDist，为节省时间和空间，此处只加入test中出现的2-gram
        for sentence in self.testset:
            if self.is_bio:  # 每一句的词频数字典
                word = bigrams(word_tokenize("BOS "+sentence+" EOS"))  # 每一句的词频数字典
            else:
                word = word_tokenize(sentence)
            for j in word:
                if j not in self.gramsDist:
                    self.gramsDist[j] = 0
                    if self.is_bio:
                        if j[0] in self.w2gram:
                            self.w2gram[j[0]] += 1
                        else:
                            self.w2gram[j[0]] = 1

    def additive_smoothing(self):  # 频数转化为频率  使用加一平滑法
        self.gramsFreq = FreqDist()  # p(w_i|w_{i-1})
        if self.is_bio:
            history = {}  # 以w为历史的2-gram的数量和
            for word in self.gramsDist:
                if word[0] in history:
                    history[word[0]] += self.gramsDist[word]
                else:
                    history[word[0]] = self.gramsDist[word]
            for word in self.gramsDist:
                self.gramsFreq[word] = (self.gramsDist[word] + 1) / (history[word[0]] + self.w2gram[word[0]])
        else:
            nv = self.gramsDist.N() + self.gramsDist.B()  # 词数总计N+词类总计V
            for word in self.gramsDist:
                self.gramsFreq[word] = (self.gramsDist[word] + 1) / nv

    def calculate_PPL(self):
        pps = []
        for sentence in self.testset:
            logppi = 0
            N = 0
            for word in (bigrams(word_tokenize("BOS "+sentence+" EOS")) if self.is_bio else word_tokenize(sentence)):
                if word in self.gramsFreq:
                    logppi += log(self.gramsFreq[word], 2) # sum(logppi)
                    N += 1
            if N > 0:
                pps.append([sentence, pow(2, -(logppi / N))])
        self.ppl = 0
        for pp in pps:
            self.ppl += pp[1]
        self.ppl /= len(pps)

    def auto_cal(self, is_bio: bool):
        self.init(is_bio=is_bio)
        self.train()
        self.add_OOV()
        self.additive_smoothing()
        self.calculate_PPL()


if __name__ == '__main__':
    gram = Gram("train_LM.txt", "test_LM.txt")
    gram.auto_cal(False)
    print("一元语法模型的困惑度:", gram.ppl)  # 885.5405291108561
    gram.auto_cal(True)
    print("二元语法模型的困惑度:", gram.ppl)  # 68.33554824217941
