# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:26:28 2017

@author: admin
"""
import jieba
import pandas as pd
#import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
#import graphviz
from sklearn.tree import DecisionTreeClassifier


data = pd.read_excel("FDATA.xlsx", sheetname='工作表1')
#newdata=np.string_((data.shape[0],2))
#newdata = [[[] for x in range(2)] for y in range(data.shape[0])]
newdata = [[] for y in range(data.shape[0])]

#Matrix = [[0 for x in range(10)] for y in range(10)]

dabang=['。','，',' ','、','的','與','（','）','；','(',')','：',
        '「','」','！','《','》','一般','為','頗','並在','']#不要的
        
for i in range(data.shape[0]):
    aa=data.ix[i,0]
    words = jieba.cut(aa, cut_all=False)
    dd=''
    for word in words:
        if(word in dabang):continue
        if(newdata[i]==[]):dd=''
        else:dd=str(newdata[i])
        ndd=dd+' '+word
        newdata[i]=ndd
    

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(newdata)
aaaaa=vectorizer.vocabulary_

tf_matrix = vectorizer.transform(newdata).toarray()
#print (tf_matrix)
#print (tf_matrix.shape)

tfidfTran = TfidfTransformer(norm="12")
tfidfTran.fit(tf_matrix)
tfidfTranidf=tfidfTran.idf_
#print (tfidfTran.idf_)


#fkdm=tfidfTranidf*tf_matrix

kmeans = KMeans(n_clusters=5, random_state=0).fit(tf_matrix)
aaaaaaaaaa=kmeans.labels_


data['mainTag'] = pd.Categorical.from_array(data.mainTag).labels #轉成數字

#--------------------
X = tf_matrix
y = data['mainTag']

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75,test_size=0.25)#前75%是訓練集、後25%當測試集
model = DecisionTreeClassifier(max_depth=100)
model = model.fit(X_train, y_train)

#dot_data = tree.export_graphviz(model, out_file=None,max_depth=None)#限制樹的深度，以免結果無法顯示
#graph = graphviz.Source(dot_data)
#graph.render("df", view=True)
#------------
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
print('Accuracy =',accuracy_score(y_test, y_predict))
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
