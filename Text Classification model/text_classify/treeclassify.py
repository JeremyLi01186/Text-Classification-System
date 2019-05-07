from sklearn.datasets import load_files
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba
from sklearn.externals import joblib
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
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
count_vect = CountVectorizer(max_features=5000)
X_train_counts = count_vect.fit_transform(files)
joblib.dump(count_vect.vocabulary_,"vocabulary.model")
# 使用tf-idf方法提取文本特征
tfidf_transformer = TfidfTransformer()
X_data = tfidf_transformer.fit_transform(X_train_counts)
# 打印特征矩阵规格
print("训练特征矩阵规格:",X_data.shape)
y=train_files.target
# parameters = {'criterion':('gini','entropy'),'splitter':('best','random'),'max_features':('auto','sqrt','log2')}
clf=tree.DecisionTreeClassifier(max_depth=4)
# clf = GridSearchCV(tree, parameters)
clf.fit(X_data,y) 
# print(clf.best_estimator_.get_params())
joblib.dump(clf, 'TREE.model')
print("模型训练完成")
