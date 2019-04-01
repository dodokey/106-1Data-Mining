# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:11:32 2018

@author: admin
"""
import pandas as pd
import numpy as np
import collections

from sklearn.model_selection import train_test_split#split for train or test data
from sklearn import tree#create tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


#------------------------讀取檔案------------------------------------------------
creditcard = pd.read_csv("UCI_Credit_Card.csv", encoding = 'ISO-8859-1')
df = creditcard.copy()
#---------------------資料前置處理-----------------------------------------------

df = df.drop('ID',axis = 1)
df.EDUCATION = df.EDUCATION.map({1:1,2:2,3:3,4:4,5:0,6:0,0:0})#0視為unknown資料
df.MARRIAGE = df.MARRIAGE.map({0:0,1:1,2:2,3:3})#0視為unknown資料



df = df.drop('BILL_AMT1',axis = 1)
df = df.drop('BILL_AMT2',axis = 1)
df = df.drop('BILL_AMT3',axis = 1)
df = df.drop('PAY_AMT1',axis = 1)
df = df.drop('PAY_AMT2',axis = 1)
df = df.drop('PAY_AMT3',axis = 1)

df = pd.get_dummies(df, columns=['SEX'])
df = pd.get_dummies(df, columns=['EDUCATION'])
df = pd.get_dummies(df, columns=['MARRIAGE'])
df = pd.get_dummies(df, columns=['PAY_0'])
df = pd.get_dummies(df, columns=['PAY_2'])
df = pd.get_dummies(df, columns=['PAY_3'])
df = pd.get_dummies(df, columns=['PAY_4'])
df = pd.get_dummies(df, columns=['PAY_5'])
df = pd.get_dummies(df, columns=['PAY_6'])


#---------------------decision tree-----------------------------------------------
X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75,test_size=0.25)#前75%是訓練集、後25%當測試集
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)


print('decision tree accuracy =',accuracy_score(y_test, y_predict))


#---------------------Kmeans-----------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeanslabels = kmeans.labels_
KmeansAcc=accuracy_score(y, kmeanslabels)

same=0
diff=0
for i in range(len(kmeanslabels)):
    if(y[i]==kmeanslabels[i]):same+=1
    else:diff+=1

if same>diff:
    print('Kmeans accuracy =',KmeansAcc)
else:#代表是反的
    print('Kmeans accuracy =',1-KmeansAcc)


#---------------------RandomForestClassifier-----------------------------------------------

rf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
rf = rf.fit(X_train, y_train)
RandomForestClassifiery_predict = rf.predict(X_test)
print('RandomForestClassifier accuracy =',accuracy_score(y_test, RandomForestClassifiery_predict))
#print('RandomForestClassifier =',scores.mean())


#---------------------10NN-----------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=20)
neigh=neigh.fit(X_train, y_train) 
neigh_predict = neigh.predict(X_test)
print('10NN accuracy =',accuracy_score(y_test, neigh_predict))





