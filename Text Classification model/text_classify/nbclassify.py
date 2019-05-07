from sklearn.datasets import load_files
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
categories=["Art","Enviornment","Agriculture","Economy","Politics","Sports","Space"]
train_files=load_files('train',
        categories=categories,
        load_content = True,
        decode_error='strict',
        shuffle=True, random_state=42)
files=[]
j=0
for data in train_files.data:
	files.append(" ".join(jieba.cut(data)))
	j+=1
# 统计词语出现次数
count_vect = CountVectorizer(max_features=20000)
X_train_counts = count_vect.fit_transform(files)
joblib.dump(count_vect.vocabulary_,"vocabulary.model")
# 使用tf-idf方法提取文本特征
tfidf_transformer = TfidfTransformer()
X_data = tfidf_transformer.fit_transform(X_train_counts)
# 打印特征矩阵规格
print("训练特征矩阵规格:",X_data.shape)
y=train_files.target
clf=MultinomialNB().fit(X_data,y) 
joblib.dump(clf, 'NB.model')
print("模型训练完成")
# print("开始测试")
# files2=[]
# for data in test_files.data:
# 	files2.append(" ".join(jieba.cut(data)))
# count_vect2=CountVectorizer(vocabulary=count_vect.vocabulary_)
# X_test_counts=count_vect2.fit_transform(files2)
# testdata=tfidf_transformer.fit_transform(X_test_counts)
# predicted=clf.predict(X_test_counts)
# print("精确率:",metrics.precision_score(test_files.target,predicted,pos_label=7,average='macro'))
# print("召回率",metrics.recall_score(test_files.target,predicted,pos_label=7, average='macro'))
# print("f-值",metrics.f1_score(test_files.target,predicted,pos_label=7, average='macro'))
# predicted = cross_val_predict(clf,X_data, y, cv=10)
# score=cross_val_score(clf, X_data, y, cv=10)
# accuracy=metrics.accuracy_score(train_files.target, predicted)
# precision=metrics.precision_score(train_files.target, predicted,pos_label=7,average='macro')
# recall=metrics.recall_score(train_files.target, predicted,pos_label=7,average='macro')
# f1=metrics.f1_score(train_files.target, predicted,pos_label=7,average='macro')
# print("贝叶斯")
# print("准确率：",accuracy)
# print("精确率：",precision)
# print("召回率：",recall)
# print("F-值:",f1)

#画图
import matplotlib.pyplot as plt
