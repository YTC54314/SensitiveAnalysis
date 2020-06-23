"""
build the model on Keras
LSTM, cross validation
Use the Keras tokenizer(tfidf)
"""

from scipy import sparse
review_sparse_matrix = sparse.load_npz("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/vectorization_5000.npz")
#print("review_sparse_matrix.shape",review_sparse_matrix.shape)
X = review_sparse_matrix
#2D轉3D(LSTM要3D matrix)
X = X.A
X = X.reshape(50000, 1, 5000)
#print("X",X)
print("X.shape",X.shape)


import pandas as pd
df = pd.read_csv("C:/Users/YT/Desktop/MLfinalproject/aclImdb_v1/cleaneddata.csv", encoding='utf-8',low_memory=False)
y = df["sentiment"]
#print("y.shape",y.shape)


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)
y.astype(int)
print(y)
print("y",y.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#print(X_train.shape)

print("X_train",X_train)
print("X_train",X_train.shape)

print("X_test",X_test)
print("X_test",X_test.shape)

print("y_train",y_train)
print("y_train",y_train.shape)

print("y_test",y_test)
print("y_test",y_test.shape)


from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(LSTM(64,input_shape=(1,5000), return_sequences=True))
model.add(Activation('relu'))
model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.1))
model.add(Activation('relu'))
model.add(Dense(2, activation='softmax'))

"""
#model.add(Embedding(35000,256,input_length=73698))
model.add(LSTM(units=128,input_shape=(35000,73698), return_sequences=True))
model.add(Dropout(0.7))
#model.add(LSTM(128,input_dim=73698,dropout=0.2))
model.add(Activation('softmax'))

model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

model.add(Dense(2))
model.add(Dropout(0.3))
model.add(Activation('softmax')) #二分類的時候直接轉換為（-1，1）
"""

#Configures the model for training.
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

#Trains the model for a given number of epochs (iterations on a dataset).
model.fit(X_train, y_train, batch_size=100, epochs=30, validation_split=0.2)

#Returns the loss value & metrics values for the model in test mode.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
print("loss:",loss)

#Generates output predictions for the input samples
result = model.predict(X_test)
print(result)