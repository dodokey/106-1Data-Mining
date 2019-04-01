# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:47:45 2017
@author: admin
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data,target=[],[]
csv = pd.read_csv("nyc-rolling-sales.csv",encoding = "ISO-8859-1")

#各欄處理
csv=csv.drop(csv.columns[0], axis=1)#0
csv = pd.get_dummies(csv, columns=['BOROUGH'])#1
csv = pd.get_dummies(csv, columns=['NEIGHBORHOOD'])#3
csv = pd.get_dummies(csv, columns=['BUILDING CLASS CATEGORY'])#3
csv = pd.get_dummies(csv, columns=['TAX CLASS AT PRESENT'])#4
#05BLOCK
#06LOT
csv=csv.drop('EASE-MENT',axis = 1)#07
csv = pd.get_dummies(csv, columns=['BUILDING CLASS AT PRESENT'])#8
csv=csv.drop('ADDRESS',axis = 1)#9
csv=csv.drop('APARTMENT NUMBER',axis = 1)#10
csv = pd.get_dummies(csv, columns=['ZIP CODE'])#11
csv=csv.drop('RESIDENTIAL UNITS',axis = 1)#12
csv=csv.drop('COMMERCIAL UNITS',axis = 1)#13
#14TOTAL UNITS
csv = csv[csv['LAND SQUARE FEET'].str.contains(" -  ") == False]#15
csv = csv[csv['GROSS SQUARE FEET'].str.contains(" -  ") == False]#16
csv=csv.drop('YEAR BUILT',axis = 1)#17
csv = pd.get_dummies(csv, columns=['TAX CLASS AT TIME OF SALE'])#18
csv = pd.get_dummies(csv, columns=['BUILDING CLASS AT TIME OF SALE'])#19
csv = csv[csv['SALE PRICE'].str.contains(" -  ") == False]#20
csv=csv.drop('SALE DATE',axis = 1)#21

csv=csv.astype(float)

m = np.percentile(csv['SALE PRICE'], 22)
M = np.percentile(csv['SALE PRICE'], 83)

csv = csv[csv['SALE PRICE'] > m]
csv = csv[csv['SALE PRICE'] < M]

data=csv.drop('SALE PRICE',axis = 1)
data=np.array(data)

target=csv[['SALE PRICE']]
target=np.array(target)
#==============================================================================
#==========================做好資料了===========================================
#==============================================================================

datadic_X = data
datadic_X_train, datadic_X_test, datadic_y_train,datadic_y_test = train_test_split(data, target,train_size=0.75,test_size=0.25)#前75%是訓練集、後25%當測試集


regr = linear_model.LinearRegression()
regr.fit(datadic_X_train, datadic_y_train)
datadic_y_pred = regr.predict(datadic_X_test)
print('----Linear Regression----')
print("Mean absolute error:" ,mean_absolute_error(datadic_y_test, datadic_y_pred))

rf = RandomForestRegressor()
rf.fit(datadic_X_train, datadic_y_train.ravel())
datadic_y_pred = rf.predict(datadic_X_test)
print('-------------------------------')
print('----Random Forest Regression----')
print("Mean absolute error:",mean_absolute_error(datadic_y_test, datadic_y_pred))