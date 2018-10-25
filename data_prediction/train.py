#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
from sklearn.externals import joblib

### read the data
df_one = pd.read_excel('event_data_0919.xls')
df_two = pd.read_excel('event_data_1011.xls')
df = pd.concat([df_one, df_two], sort = True)

### data preprocessing
df.birthday =  df.birthday.astype(str)
df['age'] = df.birthday.str.slice(0,4)
df.age = df.age.astype(int)
df.age = 2018 - df.age
data = df.drop(['data_source','data_source_person','identify_id','name','suggestion',
                'birthday','source', 'age'], axis=1)
y = data['conclusion']
x = data[data.columns.drop('conclusion')].replace(999, 0)

### predict and validation
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)
clf = lgb.LGBMClassifier(max_depth = 3)
clf.fit(train_x, train_y)
joblib.dump(clf, "train_model.m")
pred = clf.predict(test_x)
print (precision_score(test_y, pred,average='micro'))
