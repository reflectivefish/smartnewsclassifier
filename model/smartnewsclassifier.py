#!/usr/bin/env python
# coding: utf-8



import re

from sklearn.externals import joblib
from model import clean_str, vect
from textblob import Word


import PySimpleGUI as sg

import pandas as pd
#data = pd.read_csv('../dataset/dataset.csv')



#data.shape



vect = joblib.load('../model/vect.pkl')   


model = joblib.load('../model/news_classifier.pkl')





def check_news_type(news_article): 
    news_article = [' '.join([Word(word).lemmatize() for word in clean_str(news_article).split()])]
    features = vect.transform(news_article)
    #return str(model.predict(features)[0])
    output=str(model.predict(features)[0])  
    x= output[1:]
    return x

#sg.ChangeLookAndFeel('GreenTan')

text = sg.PopupGetText('CLASSIFY NEWS NOW' '')
newstype=check_news_type(text)      
sg.Popup('Results', 'The category of the news is', newstype)


text = sg.PopupGetText('CLASSIFY NEWS NOW' '')
newstype=check_news_type(text)      
sg.Popup('Results', 'The category of the news is', newstype)

text = sg.PopupGetText('CLASSIFY NEWS NOW' '')
newstype=check_news_type(text)      
sg.Popup('Results', 'The category of the news is', newstype)



