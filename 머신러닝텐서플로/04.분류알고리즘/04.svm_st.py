# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import *

#df = pd.read_csv('./preprocessed_redwine_quality.csv', engine='python', encoding='cp949')
df = pd.read_csv('C:\\AI_CTF\\ch03\\04.분류알고리즘\\preprocessed_redwine_quality.csv', engine='python', encoding='cp949')

train, test = train_test_split(df, test_size=0.3) 

train_x = train.drop('품질수준', 1)    # 1: col
train_y = train['품질수준']
test_x = test.drop('품질수준', 1)
test_y = test['품질수준']

kernel_list = ['linear', 'rbf', 'sigmoid']
C_list = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]

for kernel in kernel_list:
    for C in C_list:
        model = svm.SVC(C=C, kernel=kernel).fit(train_x, train_y)
        predited = model.predict(test_x)
        print(kernel, C, accuracy_score(test_y, predited))
        

#linear 0.03125 0.8210526315789474
#linear 0.125 0.8210526315789474
#linear 0.5 0.8210526315789474
#linear 2 0.8210526315789474
#linear 8 0.8210526315789474
#linear 32 0.8210526315789474
#linear 128 0.8210526315789474
#linear 512 0.8294736842105264
#linear 2048 0.8294736842105264
#linear 8192 0.8273684210526315
#linear 32768 0.8378947368421052
#rbf 0.03125 0.8210526315789474
#rbf 0.125 0.8210526315789474
#rbf 0.5 0.8189473684210526
#rbf 2 0.8189473684210526
#rbf 8 0.8042105263157895
#rbf 32 0.8021052631578948
#rbf 128 0.7978947368421052
#rbf 512 0.7978947368421052
#rbf 2048 0.7978947368421052
#rbf 8192 0.7978947368421052
#rbf 32768 0.7978947368421052
#sigmoid 0.03125 0.8210526315789474
#sigmoid 0.125 0.8210526315789474
#sigmoid 0.5 0.8210526315789474
#sigmoid 2 0.8210526315789474
#sigmoid 8 0.8210526315789474
#sigmoid 32 0.8210526315789474
#sigmoid 128 0.8210526315789474
#sigmoid 512 0.8210526315789474
#sigmoid 2048 0.8210526315789474
#sigmoid 8192 0.8210526315789474
#sigmoid 32768 0.8210526315789474


