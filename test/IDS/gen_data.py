import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def shuffle_and_batch(data, labels, batch_size):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return data[:batch_size], labels[:batch_size]


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


def z_normalisation(data):
    data = preprocessing.scale(data)
    return data, data.mean, data.std


def one_hot_coding(y, num_classes=None):
   # print('data',data)
   # data = keras.utils.to_categorical(data, num_classes=15)
   # print(data)

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    print(num_classes)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

