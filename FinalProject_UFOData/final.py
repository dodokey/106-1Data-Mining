# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:14:43 2017

@author: ORLab
"""

import pandas as pd
from sklearn.model_selection import train_test_split#split for train or test data
from sklearn import tree#create tree
import decimal
from sklearn.metrics import accuracy_score
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#------------------------讀取檔案------------------------------------------------
df = pd.read_csv("complete.csv", encoding = 'ISO-8859-1')
#---------------------資料前置處理-----------------------------------------------
delete = []
df = df.fillna(0)
for i in range(len(df)):
    if(df['country'][i] != 'us' or df['country'][i] == 0):
        delete.append(i)
df = df.drop(df.index[delete])
   
#--------------------建立MODEL判斷州---------------------------------------------
X = df.drop(df.columns[[1,2,3,4,6,7,8]], axis=1)
y = df['state']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)#75% for training data,remain to test data
model = tree.DecisionTreeClassifier()#create tree
model = model.fit(X_train, y_train)#train it
y_predict = model.predict(X_test)
#print(y_predict +' '+ y_test)
print("Accuracy: " + str(round(decimal.Decimal(accuracy_score(y_test, y_predict)*100),3)) + '%')


#-------------------取COMMENTS關鍵字---------------------------------------------
stateArr = set()
commentarr = []
for i in df.index:
    stateArr.add(df['state'][i])
stateArr = list(stateArr)
for i in range(len(stateArr)):
    tmp = ""
    for j in df.index:
        if df['state'][j] == stateArr[i]:
            tmp += str(df['comments'][j])
    commentarr.append([stateArr[i],tmp])
commentarr = np.array(commentarr)
keywordarr = []
delwords = [' ','','　','&#','44','is','are','was','were','has','have','my','。','，','(',')','/','《','》','「','」','!','．','（','）','、'] # 刪除沒有意義的文字
for i in range(commentarr.shape[0]):
    words = jieba.cut(commentarr[i][1], cut_all=False)
    tmp = ''
    for word in words:
        if(word not in delwords):
            tmp += word+' '
    commentarr[i][1] = tmp
    keyword = jieba.analyse.extract_tags(commentarr[i][1], 10)
    keywordarr.append([commentarr[i][0],keyword])

#----------------------使用者輸入------------------------------------------------

testdata=df[['datetime','duration (seconds)','latitude','longitude','state','comments']][int(len(df)*0.75):-1]

test=testdata.drop(testdata[['state','comments']],axis=1)
testans=testdata['state']
commentss=np.array(testdata['comments'])


predict_result = model.predict(test)

stateaccuracy=accuracy_score(testans, predict_result)
print('判斷州的正確率',stateaccuracy)


wordss = jieba.cut(commentss, cut_all=False)

true=0

for x in range(len(commentss)):
    tmp = 0
    user_input_comment=commentss[x]
    for i in range(len(keywordarr)):
        if(keywordarr[i][0] == predict_result[x]):
            words = jieba.cut(user_input_comment, cut_all=False)
            keyword_tmp = []
            for word in words:
                if(word in keywordarr[i][1]):
                    if(word not in keyword_tmp):
                        tmp += 1
                        keyword_tmp.append(word)
                    
#    print('Your infomation\'s reliability is', tmp/10*100, '%')
    if tmp/10*100 >= 10.0:
#        print('it can be trusted ! ')
        true+=1
    else:
        continue
#        print('fake !')


print('True or Fake Accuracy:',true/len(commentss))








