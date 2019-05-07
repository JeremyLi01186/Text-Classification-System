from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.externals import joblib
import jieba
def calculate_result(actual,pred):
	m_precision=metrics.precision_score(actual,pred,pos_label=7, average='macro')
	m_recall=metrics.recall_score(actual,pred,pos_label=7, average='macro')
	print("预测结果:")
	print("准确率:",m_precision)
	print("召回率:",m_recall)
	print("F-值:",2*m_precision*m_recall/(m_precision+m_recall))
categories=["Art","Enviornment","Agriculture","Economy","Politics","Sports","Space"]
test_files=load_files('test',
        categories=categories,
        load_content = True,
        decode_error='strict',
        shuffle=True, random_state=42)
print("测试文件数目:",len(test_files.data))

files=[]
for data in test_files.data:
	files.append(" ".join(jieba.cut(data)))
# 加载词库
vocabularyModel = joblib.load('vocabulary.model')
count_vect=CountVectorizer(vocabulary=vocabularyModel)
X_test_counts = count_vect.fit_transform(files)
# 使用tf-idf方法提取文本特征
tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
clf = joblib.load('TREE.model')
#预测结果向量 
predicted =clf.predict(X_test_tfidf)
print(metrics.classification_report(test_files.target,predicted,target_names=test_files.target_names))
print(metrics.confusion_matrix(test_files.target, predicted))
j=0
for result in predicted:
	print("文件",j+1,"真实类别:",categories[test_files.target[j]],"=========>","分类结果:",categories[result])
	j+=1
