#!/usr/bin/env python
# coding=UTF-8
"""
@version: python3.6.1
@author: TsungHan Yu
@file: article_bunch.py
@time: 2017/5/15
@software: PyCharm
"""
import os
import _pickle as pickle
from sklearn.datasets.base import Bunch


def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def corpus2Bunch(wordbag_path, seg_path):
    # 獲取seg_path下的所有子目錄
    catelist = os.listdir(seg_path)

    # 創建Bunch實例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

    bunch.target_name.extend(catelist)

    # 每個子目錄下的所有文章
    for mydir in catelist:

        # 所有以分詞的目錄路徑
        class_path = seg_path + mydir + "/"

        if mydir == '.DS_Store':
            os.remove(seg_path + mydir)

        if mydir[0] == '.':
            print(mydir)
            continue

        # 目錄下所有文章名字的陣列
        file_list = os.listdir(class_path)
        for file_path in file_list:

            if file_path == '.DS_Store':
                os.remove(class_path + file_path)

            if file_path[0] == '.':
                print(file_path)
                continue

            # 拼出文件名全路徑
            fullname = class_path + file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            # 讀取文件內容
            bunch.contents.append(readfile(fullname))

    # 把bunch保存在wordbag_path路徑中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)

    print("構建bunch完成")


if __name__ == "__main__":
    # 對訓練集Bunch化操作：
    # Bunch保存路徑
    wordbag_path = "../data/train_word_bag/train_set.dat"
    seg_path = "../data/train_corpus_seg/"
    corpus2Bunch(wordbag_path, seg_path)

    # 對測試集Bunch化操作：
    # Bunch保存路徑
    wordbag_path = "../data/test_word_bag/test_set.dat"
    seg_path = "../data/test_corpus_seg/"
    corpus2Bunch(wordbag_path, seg_path)
