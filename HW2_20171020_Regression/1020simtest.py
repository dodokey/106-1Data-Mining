# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:12:07 2017

@author: admin
"""


# We are gonna use Scikit's LinearRegression model
from sklearn.linear_model import LinearRegression


# Your input data, X and Y are lists (or Numpy Arrays)
x = [[2,4],[3,6],[4,5],[6,7],[3,3],[2,5],[5,2],[4,5],[8,8],[9,8]]
y = [8,18,20,42,9,10,10,20,64,72]

# Initialize the model then train it on the data
genius_regression_model = LinearRegression()
genius_regression_model.fit(x,y)

# Predict the corresponding value of Y for X = [8,4]
print ('heyyy:::',genius_regression_model.predict([5,6]))

