# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:52:41 2017

@author: admin
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
#from sklearn import linear_model
#from sklearn.metrics import mean_absolute_error
#from sklearn.ensemble import RandomForestRegressor

csv = pd.read_csv("data.csv",encoding = "ISO-8859-1")
#各欄處理
#age
csv = pd.get_dummies(csv, columns=['workclass'])#2
#fnlwgt
csv = pd.get_dummies(csv, columns=['education'])#4
csv = pd.get_dummies(csv, columns=['education_num'])#5
csv = pd.get_dummies(csv, columns=['marital_status'])#6
csv = pd.get_dummies(csv, columns=['occupation'])#7
csv = pd.get_dummies(csv, columns=['relationship'])#8
csv = pd.get_dummies(csv, columns=['race'])#9
csv = pd.get_dummies(csv, columns=['sex'])#10
csv = pd.get_dummies(csv, columns=['capital_gain'])#11
csv = pd.get_dummies(csv, columns=['capital_loss'])#12
#hour_per_week
#csv = csv[csv['native_country'].str.contains(" ?") == False]#14
csv = pd.get_dummies(csv, columns=['native_country'])#12
#csv=csv.drop('native_country',axis = 1)#11
csv['income'] = np.where(csv.income.isin([' <=50K']),'0', csv['income'])# <=50K=0
csv['income'] = np.where(csv.income.isin([' >50K']),'1', csv['income'])#  >50K=1


data=csv.drop('income',axis = 1)
data=np.array(data)

target=csv[['income']]
target=np.array(target)
#dataok========================================================================

#==============================================================================
# def K_fold_CV(k, data,target):
#     kf = KFold(n_splits=k)
#     Accuracy=0
#     
#     for train_index, test_index in kf.split(data):
#         X_train, X_test = data[train_index], data[test_index]
#         Y_train, Y_test = target[train_index], target[test_index]
#         regr = GradientBoostingClassifier()
#         regr.fit(X_train, Y_train)
#         Y_pred = regr.predict(X_test)
#         thisacc=accuracy_score(Y_test, Y_pred)
#         Accuracy+=thisacc
#         print('accuracy=',thisacc)
#  
#     return Accuracy/k
#==============================================================================


def hw_K_fold(k, data,target):
    ksize=data.shape[0]//k
    Accuracy=0
    nowindex=0
    for i in range(k):
        X_train= np.concatenate((data[0:nowindex],data[nowindex+ksize+1:]), axis=0)
        X_test = data[nowindex:nowindex+ksize+1]
        Y_train= np.concatenate((target[0:nowindex],target[nowindex+ksize+1:]), axis=0)
        Y_test = target[nowindex:nowindex+ksize+1]
        
        
        regr = GradientBoostingClassifier()
        regr.fit(X_train, Y_train)
        Y_pred = regr.predict(X_test)
        thisacc=accuracy_score(Y_test, Y_pred)
        Accuracy+=thisacc
        print('accuracy',i+1,'=',thisacc)
        nowindex+=ksize
 
    return Accuracy/k


#print(K_fold_CV(10, data,target))
print(hw_K_fold(10, data,target))

#==============================================================================
# from sklearn.model_selection import KFold
#     X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
#     y = np.array([1, 2, 3, 4])
#     kf = KFold(n_splits=2)
#     kf.get_n_splits(X)
# 
#     for train_index, test_index in kf.split(X):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#==============================================================================



#-----------------------------
#==============================================================================
# X_train, X_test, y_train,y_test = train_test_split(data, target,train_size=0.75,test_size=0.25)#前75%是訓練集、後25%當測試集
# 
# 
# regr = GradientBoostingClassifier()
# regr.fit(X_train, y_train)
# y_pred = regr.predict(X_test)
# 
# print('accuracy =',accuracy_score(y_test, y_pred))
# 
#==============================================================================



