#!/usr/bin/env python
# coding=UTF-8
"""
@version: python3.6.1
@author: TsungHan Yu
@file: NBayes_Predict.py
@time: 2017/5/16
@software: PyCharm
"""

import _pickle as pickle

from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


trainpath = "../data/train_word_bag/tfdifspace.dat"
train_set = _readbunchobj(trainpath)

testpath = "../data/test_word_bag/testspace.dat"
test_set = _readbunchobj(testpath)

clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)
joblib.dump(clf, '../save/clf.pkl')
clf = joblib.load('../save/clf.pkl')

predicted = clf.predict(test_set.tdm)

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 實際類別:", flabel, " --> 預測類別:", expct_cate)

print("預測完成")


def metrics_result(actual, predict):
    print('precision:{0:.5f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('recall:{0:0.5f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.5f}'.format(metrics.f1_score(actual, predict, average='weighted')))


metrics_result(test_set.label, predicted)
