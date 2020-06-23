"""
build the model on Keras
ANN, cross validation
Use the Keras tokenizer(tfidf)
"""

from scipy import sparse
#review_sparse_matrix = sparse.load_npz("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/vectorization.npz")
review_sparse_matrix = sparse.load_npz("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/vectorization_1000.npz")
#print("review_sparse_matrix.shape",review_sparse_matrix.shape)
X = review_sparse_matrix

import pandas as pd
df = pd.read_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/cleaneddata.csv", encoding='utf-8',low_memory=False)
y = df["sentiment"]
#print("y.shape",y.shape)
"""
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)
y.astype(int)
print(y)
print("y",y.shape)
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""
print("X_train",X_train)
print("X_train",X_train.shape)

print("X_test",X_test[0])
print("X_test",X_test[0].shape)
print("X_test",X_test[1])
print("X_test",X_test[1].shape)
print("X_test",X_test.shape)

print("y_train",y_train)
print("y_train",y_train.shape)

print("y_test",y_test)
print("y_test",y_test.shape)
"""


#把數據分割為training data和test data

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation

model=Sequential()
#model.add(Dense(128, input_dim = 73698))
model.add(Dense(128, input_dim = 1000))
model.add(Dropout(0.4))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(Dropout(0.7))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))



#Configures the model for training.
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())




#Trains the model for a given number of epochs (iterations on a dataset).
model.fit(X_train, y_train, batch_size=100, epochs=10, validation_split=0.2)




#Returns the loss value & metrics values for the model in test mode.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
print("loss:",loss)



import numpy as np
#Generates output predictions for the input samples
result = model.predict(X_test)
print(result)
print(result.shape)
result = np.rint(result)

print("the indexs of incorrect predictions:")
count=0
for i in range(0,14999):
    if (y_test.values[i] != result[i]):
        print(y_test.index[i])
        count+=1
		
print("numbers of incorrect predictions",count)
print("percetage of incorrect predictions",round(((count/15000)*100),2))

"""
import matplotlib.pyplot as plt
plt.plot(y_test.value[i],'ro',label="real")
plt.plot(y_hat[i],'bs',label="prediction")

plt.xlabel('review')
plt.ylabel('result')
plt.show()
"""

