
import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
from sklearn import svm, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam


import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

import datetime as dt
import math
import time
import keras
# Generate dummy data

# SVM


def SVM(x_train, y_train, x_test, y_test):
    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C, gamma=param_gamma)

    start_time = dt.datetime.now()
    print(start_time)
    print('Start learning at {}'.format(str(start_time)))
    classifier.fit(x_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    expected = y_test
    predicted = classifier.predict(x_test)

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


# MLP
def MLP():
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(256, activation='relu', input_dim=78))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    sgd = SGD(lr=0.015, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


# load data
def loaddata():
    train = pd.read_csv(
        "~/PycharmProjects/data/TrafficLabelling/train_set.csv",
        encoding="ISO-8859-1",
        low_memory=False)
    test = pd.read_csv(
        "~/PycharmProjects/data/TrafficLabelling/test_set.csv",
        encoding="ISO-8859-1",
        low_memory=False)
    crossval = pd.read_csv(
        "~/PycharmProjects/data/TrafficLabelling/crossval_set.csv",
        encoding="ISO-8859-1",
        low_memory=False)
    return train, test, crossval


def changelabel(train, test, crossval):
    le = preprocessing.LabelEncoder()
    le.fit(['BENIGN',
            'FTP-Patator',
            'SSH-Patator',
            'DoS Hulk',
            'DoS GoldenEye',
            'DoS slowloris',
            'DoS Slowhttptest',
            'Heartbleed',
            'Web Attack \x96 Brute Force',
            'Web Attack \x96 XSS',
            'Web Attack \x96 Sql Injection',
            'Infiltration',
            'Bot',
            'PortScan',
            'DDoS'])
    train[' Label'] = le.transform(train[' Label'])
    test[' Label'] = le.transform(test[' Label'])
    crossval[' Label'] = le.transform(crossval[' Label'])
    return train, test, crossval


def main():
    train, test, crossval = loaddata()
    train, test, crossval = changelabel(train, test, crossval)

    # normalization

    train = preprocessing.scale(train)
    test = preprocessing.scale(test)

    X_TRAIN = train[:, 1:79]
    Y_TRAIN = train[:, 79]
    X_TEST = test[:, 1:79]
    Y_TEST = test[:, 79]
    '''
    X_TRAIN = X_TRAIN.ix[0:10000, :]
    Y_TRAIN = Y_TRAIN.ix[0:10000]
    X_TEST = X_TEST.ix[0:10000,:]
    Y_TEST = Y_TEST.ix[0:10000]
    '''
    Y_TRAIN = keras.utils.to_categorical(Y_TRAIN, num_classes=15)
    Y_TEST = keras.utils.to_categorical(Y_TEST, num_classes=15)

    model = MLP()
    model.fit(X_TRAIN, Y_TRAIN,
              epochs=3,
              batch_size=128)
    score = model.evaluate(X_TEST, Y_TEST, batch_size=128)
    print(score)

    #SVM(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)


if __name__ == "__main__":
    main()
