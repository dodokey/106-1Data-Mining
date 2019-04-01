# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:19:43 2017

@author: admin
"""
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#import graphviz
from sklearn.tree import DecisionTreeClassifier


data=pd.read_excel("FDATA.xlsx")
cutdata=[]
for i in range(data.shape[0]):
    x = jieba.analyse.extract_tags(data["postContent"][i], 200)
    cutdata.append(' '.join(x))



vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(cutdata))
word = vectorizer.get_feature_names()
datatfidf = tfidf.toarray()


kmeans = KMeans(n_clusters=5, random_state=0).fit(datatfidf)
kmeanslabels = kmeans.labels_
labelsslabel=[[],[],[],[],[]]

for i in range(len(kmeanslabels)):
    labelsslabel[kmeanslabels[i]].append(data['mainTag'][i])

allcorrect=0
for i in range(5):
    print('第',i+1,'群有',len(labelsslabel[i]),'篇文章。')
    setlabel=list(set(labelsslabel[i]))
    y=np.zeros(len(setlabel))
    for x in labelsslabel[i]:
        y[setlabel.index(x)]+=1
    print('包含了：')
    for z in range(len(setlabel)):
        print(int(y[z]),'篇為',setlabel[z],'分類。', end='')
    print()
    print('判斷此群為',setlabel[np.argmax(y)])
    print('正確率為：',max(y)/sum(y))
    allcorrect=allcorrect+max(y)
    print()
print('xxxxxxxxxxxxxxxxxxxxxxx')
print('k-means總正確率:',allcorrect/len(data))
print('xxxxxxxxxxxxxxxxxxxxxxx')


#print (set(labelsslabel[0]))


data['mainTag'] = pd.Categorical.from_array(data.mainTag).labels
X = datatfidf
Y = data['mainTag']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
clf=DecisionTreeClassifier(max_depth=10)
clf=clf.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
Y_predict=clf.predict(X_test)
print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
print("Accuracy=",accuracy_score(Y_test,Y_predict))
print('xxxxxxxxxxxxxxxxxxxxxxxxxx')


#==============================================================================
# 
# #featuress=arr_newdata[:,0]
# #targetss=arr_newdata[:,1]
# #
# #transformer = TfidfTransformer(smooth_idf=False)
# #tfidf = transformer.fit_transform(featuress)
# 
# stemmer = nltk.stem.porter.PorterStemmer()
# def StemTokens(tokens):
#     return [stemmer.stem(token) for token in tokens]
# remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# def StemNormalize(text):
#     return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
# 
# 
# lemmer = nltk.stem.WordNetLemmatizer()
# def LemTokens(tokens):
#     return [lemmer.lemmatize(token) for token in tokens]
# remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# def LemNormalize(text):
#     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
# 
# 
# 
# LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
# LemVectorizer.fit_transform(newdatalist)
# 
# print (LemVectorizer.vocabulary_)
# 
# 
# tf_matrix = LemVectorizer.transform(notcutlist).toarray()
# print (tf_matrix)
# print (tf_matrix.shape)
# 
# 
# 
# tfidfTran = TfidfTransformer(norm="l2")
# tfidfTran.fit(tf_matrix)
# tfidfTranidf=tfidfTran.idf_
# print (tfidfTran.idf_)
# 
# 
# #==============================================================================
# # tfidf_matrix = tfidfTran.transform(tf_matrix)
# # #print (tfidf_matrix.toarray())
# # 
# # cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
# # print (cos_similarity_matrix)
# #==============================================================================
# 
# 
# fkdm=tfidfTranidf*tf_matrix
# 
# kmeans = KMeans(n_clusters=5, random_state=0).fit(fkdm)
# aaaaa=kmeans.labels_
# 
#==============================================================================
