from crf import CRF

import sys
path1 = r"D:\programs\course\nlplab\exp3\CRF实现词性标注实验\bi-lstm-crf-master"
sys.path.append(path1)
path2 = r"D:\programs\course\nlplab\exp3\CRF实现词性标注实验\keras-contrib-master"
sys.path.append(path2)
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from dl_segmenter import get_or_create, save_config,DLSegmenter
from dl_segmenter.custom.callbacks import LRFinder, SGDRScheduler, WatchScheduler
from dl_segmenter.data_loader import DataLoader
from dl_segmenter.utils import make_dictionaries
import os
import re

if __name__ == '__main__':
    crf = CRF()