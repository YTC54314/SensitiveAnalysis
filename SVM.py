from scipy import sparse
review_sparse_matrix = sparse.load_npz("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/vectorization_1000.npz")
X = review_sparse_matrix

import pandas as pd
df = pd.read_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/cleaneddata.csv", encoding='utf-8',low_memory=False)
y = df["sentiment"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("X_train",X_train)
print("X_train",X_train.shape)

print("y_train",y_train)
print("y_train",y_train.shape)

print("y_test",y_test)
print("y_test",y_test.shape)

from sklearn import svm
# "C" -> A large value of c means you will get more training points correctly.
#"gamma" -> It defines how far the influence of a single training example reaches.
#if the gamma value is low even the far away points get considerable weight and we get a more linear curve.
#decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')

#clf = svm.SVC(kernel="linear", C=0.8, gamma=20, decision_function_shape='ovr')
# SVC(kernel="linear", C=0.025),
clf.fit(X_train, y_train)
print("222222222222222222222222")

print (clf.score(X_train, y_train))
y_hat = clf.predict(X_train)
print(y_hat)

print (clf.score(X_test, y_test))
y_hat = clf.predict(X_test)
print(y_hat)

from sklearn.externals import joblib #jbolib模块

#保存Model
joblib.dump(clf, 'C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/clf.pkl')
"""
#读取Model
clf3 = joblib.load('save/clf.pkl')

#测试读取后的Model
print(clf3.predict(X[0:1]))
"""