# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import math

#df = pd.read_csv('./preprocessed_redwine_quality.csv', engine='python', encoding='cp949')
df = pd.read_csv('C:\\AI_CTF\\ch03\\04.분류알고리즘\\preprocessed_redwine_quality.csv', engine='python', encoding='cp949')

train, test = train_test_split(df, test_size=0.3) 

train_x = train.drop('품질수준', 1)    # 1: col
train_y = train['품질수준']
test_x = test.drop('품질수준', 1)
test_y = test['품질수준']

M = int(math.sqrt(len(train_x)))
for k in range(1, M, 2):
     #n_neighbors=5(default), algorithm='auto'(default), metric='minkowski’(default)
    model = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree').fit(train_x, train_y)
    predited = model.predict(test_x)
    print(k, accuracy_score(test_y, predited))

#1 0.8
#3 0.8042105263157895
#5 0.8147368421052632
#7 0.8231578947368421
#9 0.8168421052631579
#11 0.8273684210526315
#13 0.8421052631578947
#15 0.8421052631578947
#17 0.8357894736842105
#19 0.8336842105263158
#21 0.84
#23 0.8336842105263158
#25 0.84
#27 0.8421052631578947
#29 0.8421052631578947
#31 0.8421052631578947

