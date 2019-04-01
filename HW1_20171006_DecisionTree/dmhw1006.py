# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:07:11 2017

@author: admin
"""
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("character-deaths.csv")#讀取資料
df=df.fillna(0)#把空值以0替代
df['Death Year'][df['Death Year'] != 0] = 1#Death Year將有數值的轉成1
df=df.drop('Book of Death',axis = 1)
df=df.drop('Death Chapter',axis = 1)
df = pd.get_dummies(df, columns=['Allegiances'])#將Allegiances底下的家族轉成dummy的特徵
df = df.drop('Name',axis = 1)
#------------
X = df.drop('Death Year', axis=1)
y = df['Death Year']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75,test_size=0.25)#前75%是訓練集、後25%當測試集
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)

dot_data = tree.export_graphviz(model, out_file=None,max_depth=6)#限制樹的深度，以免結果無法顯示
graph = graphviz.Source(dot_data)
graph.render("df", view=True)
#------------
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
print('Precision Rate =',precision_score(y_test, y_predict))
print('Recall Rate =',recall_score(y_test, y_predict))
print('accuracy =',accuracy_score(y_test, y_predict))
