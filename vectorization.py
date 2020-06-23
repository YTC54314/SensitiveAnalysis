"""
vectorization
CountVectorizer+TfidfTransformer
https://www.cnblogs.com/CheeseZH/p/8644893.html
"""


import pandas as pd
df = pd.read_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/cleaneddata.csv", encoding='utf-8',low_memory=False)
corpus = df['review']

from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(max_features = 5000)
vectorizer = CountVectorizer(max_features = 1000)
#根据语料集(文本)统计词袋（fit）
#統整文本單字(把重複單字歸納為一個單字),形成詞帶
vectorizer.fit(corpus)
#将语料集转化为词袋向量（transform）
X = vectorizer.transform(corpus)


#將CountVectorizer結果,依據出現頻率計算tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
#根据语料集的词袋向量计算TF-IDF（fit）
tfidf_transformer.fit(X)
#将语料集的词袋向量表示转换为TF-IDF向量表示(transform)
X_csr_matrix = tfidf_transformer.transform(X)
print(type(X_csr_matrix))
print(X_csr_matrix.shape)
print(X_csr_matrix)
#print(X_csr_matrix[0][:100].toarray())

#import numpy
#numpy.savetxt("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/vectorization1", X_csr_matrix[0][:100].toarray())

from scipy import sparse
sparse.save_npz("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/vectorization_1000.npz", X_csr_matrix)

"""
print("start csving.")
df.columns = ["review","sentiment"]
df.to_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/vectorization.csv",index=False)
print("finish.")"""