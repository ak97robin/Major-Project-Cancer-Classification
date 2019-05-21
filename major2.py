#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:03:28 2019

@author: atish
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset=pd.read_csv('data.csv')
dataset.head()
x=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,1].values

#cleaning data
print("Cancer"+format(dataset.shape))
dataset.isnull().sum()
dataset.isna().sum()

#encoding malignant and benign into binary data
from sklearn.preprocessing import LabelEncoder
labelencode_y=LabelEncoder()
y=labelencode_y.fit_transform(y)
print(y)

#splitting dataset
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
trainx=sc.fit_transform(trainx)
testx=sc.transform(testx)

#fitting model
from sklearn.neighbours import KN