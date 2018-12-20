
import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
# Generate dummy data

#MLP
def MLP():
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(256, activation='relu', input_dim=78))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    return model

#load data
def loaddata():
    train = pd.read_csv("~/PycharmProjects/test/data/TrafficLabelling/train_set.csv", encoding="ISO-8859-1", low_memory=False)
    test = pd.read_csv("~/PycharmProjects/test/data/TrafficLabelling/test_set.csv", encoding="ISO-8859-1", low_memory=False)
    crossval = pd.read_csv("~/PycharmProjects/test/data/TrafficLabelling/crossval_set.csv", encoding="ISO-8859-1", low_memory=False)
    return train,test,crossval

def changelabel(train,test,crossval):
    le = preprocessing.LabelEncoder()
    le.fit(['BENIGN', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris','DoS Slowhttptest',
            'Heartbleed','Web Attack \x96 Brute Force','Web Attack \x96 XSS','Web Attack \x96 Sql Injection',
            'Infiltration','Bot','PortScan','DDoS'])
    train[' Label'] = le.transform(train[' Label'])
    test[' Label'] = le.transform(test[' Label'])
    crossval[' Label'] = le.transform(crossval[' Label'])
    return train,test,crossval

def main():
    train,test,crossval = loaddata()
    train, test, crossval = changelabel(train,test,crossval)
    X_TRAIN = train.ix[:,1:79]
    Y_TRAIN = train.ix[:,79]
    X_TEST = test.ix[:,1:79]
    Y_TEST = test.ix[:,79]
    Y_TRAIN = keras.utils.to_categorical(Y_TRAIN, num_classes=15)
    Y_TEST = keras.utils.to_categorical(Y_TEST, num_classes=15)
    model = MLP()
    model.fit(X_TRAIN, Y_TRAIN,
              epochs=20,
              batch_size=128)
    score = model.evaluate(X_TEST, Y_TEST, batch_size=128)

if __name__ == "__main__":
    main()