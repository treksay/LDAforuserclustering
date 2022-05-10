# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:28:13 2022

@author: Tarek
"""

import pandas as pd
dataset = pd.read_csv('dataset.csv')
dataset = dataset[0:11000]
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = cv.fit_transform(dataset['text'])

from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=20,random_state=42)
LDA.fit(dtm)
len(cv.get_feature_names())
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

c = LDA.transform(dtm)
    