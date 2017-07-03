#!/usr/bin/env python
# coding=UTF-8
"""
@version: python3.6.1
@author: TsungHan Yu
@file: classification.py
@time: 2017/5/31
@software: PyCharm
"""
import os

import jieba
import _pickle as pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn import metrics

# 將內容保存到 `save_path`
from sklearn.naive_bayes import MultinomialNB


def save_file(save_path, content):
    with open(save_path, "w", encoding='utf8') as fp:
        fp.write(content)


# 讀取文件
def read_file(path):
    with open(path, "r", encoding='utf8') as fp:
        content = fp.read()
    return content


def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 將文章做分詞
# article_path  -> 未分詞文章路徑
# seg_path      -> 已分詞文章路徑
def article_segment(article_path, seg_path):
    # os.listdir() 方法返回指定的文件夾包含的文件或文件夾的名字的列表 ['.DS_Store', 'Art', 'Education', 'Literature']
    catelist = os.listdir(article_path)

    # for-in 目錄下的文件
    for file_path in catelist:

        if file_path == '.DS_Store':
            os.remove(article_path + file_path)

        if file_path[0] == '.':
            print('###', file_path)
            continue

        # 文章路徑  EX: train_corpus/art/21.txt
        fullname = article_path + file_path
        # 文章內容
        content = read_file(fullname)
        # 刪除換行
        content = content.replace('\n', '')
        content = content.replace('\r', '')
        # 刪除空格、空行
        content = content.replace(' ', '')
        # 使用 jieba 做分詞
        content_seg = jieba.cut(content)
        # 將分詞完的內容保存
        save_file(seg_path + file_path, " ".join(content_seg))

    print("完成分詞")


def corpus2Bunch(wordbag_path, seg_path):
    # 獲取seg_path下的所有子目錄
    catelist = os.listdir(seg_path)

    # 創建Bunch實例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

    bunch.target_name.extend(catelist)


    for file_path in catelist:

        if file_path == '.DS_Store':
            os.remove(seg_path + file_path)

        if file_path[0] == '.':
            print(file_path)
            continue

        # 拼出文件名全路徑
        fullname = seg_path + file_path
        bunch.label.append('unknown')
        bunch.filenames.append(fullname)
        # 讀取文件內容
        bunch.contents.append(read_file(fullname))


    # 把bunch保存在wordbag_path路徑中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)

print("構建bunch完成")


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    stpwrdlst = read_file(stopword_path).splitlines()
    bunch = _readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = _readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    _writebunchobj(space_path, tfidfspace)
    print("if-idf詞向量空間實例成功")


# delete files and folders
def delete_file_folder(src):
    if os.path.isfile(src):
        try:
            os.remove(src)
        except:
            pass
    elif os.path.isdir(src):
        for item in os.listdir(src):
            itemsrc = os.path.join(src, item)
            delete_file_folder(itemsrc)
        try:
            os.rmdir(src)
        except:
            pass


if __name__ == '__main__':

    if not os.path.exists('../unknown_word_bag'):
        os.makedirs('../unknown_word_bag')
    if not os.path.exists('../unknown_seg'):
        os.makedirs('../unknown_seg')
    # 未分詞分類語料庫路徑
    article_path = "../unknown/"
    # 分詞後分類語料庫路徑
    seg_path = "../unknown_seg/"
    article_segment(article_path, seg_path)

    # 對測試集Bunch化操作：
    # Bunch保存路徑
    wordbag_path = "../unknown_word_bag/unknown_set.dat"
    corpus2Bunch(wordbag_path, seg_path)

    stopword_path = "../data/train_word_bag/hlt_stop_words.txt"
    space_path = "../unknown_word_bag/unknownspace.dat"
    train_tfidf_path = "../data/train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, wordbag_path, space_path, train_tfidf_path)

    train_set = _readbunchobj(train_tfidf_path)
    test_set = _readbunchobj(space_path)

    clf = joblib.load('../save/clf.pkl')

    predicted = clf.predict(test_set.tdm)

    for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
        if flabel != expct_cate:
            print(file_name, " -> 狀態 :", flabel, " -> 預測類別:", expct_cate)

    print("預測完成")
    delete_file_folder('../unknown_word_bag')
    delete_file_folder('../unknown_seg')
