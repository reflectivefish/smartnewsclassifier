#!/usr/bin/env python
# coding: utf-8

import re

from sklearn.externals import joblib
from model import clean_str, vect
from textblob import Word


import pandas as pd
data = pd.read_csv('../dataset/dataset.csv')


def clean_str(string):
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

data.shape


vect = joblib.load('../model/vect.pkl')   
model = joblib.load('../model/news_classifier.pkl') 
int_article = clean_str(str(input()))

def check_news_type(news_article):  
    news_article = [' '.join([Word(word).lemmatize() for word in clean_str(news_article).split()])]
    features = vect.transform(news_article)
    return str(model.predict(features)[0])
out=check_news_type(int_article)
print(out)
