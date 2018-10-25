#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
from sklearn.externals import joblib

clf = joblib.load("train_model.m")
data = pd.DataFrame([[-1.25, 2.50, 175, 54, -1.00, 2.50, 172, 0.6, 0.6]], columns =
                    ['L_DC', 'L_DS', 'L_axis', 'PD', 'R_DC', 'R_DS', 'R_axis', 'left_eye', 'right_eye'])
result = str(clf.predict(data))[2:-2]
print(result)
