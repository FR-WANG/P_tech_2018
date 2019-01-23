import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
import tensorflow as tf

def shuffle(data, labels, batch_size):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    batch_number = int(len(data) / batch_size)

    return data, labels, batch_number

def batch(data, label, batch_num, batchSize):
    data_batch = data[(batch_num * batchSize): ((batch_num + 1) * batchSize) - 1]
    label_batch = label[(batch_num * batchSize): ((batch_num + 1) * batchSize) - 1]
    return data_batch, label_batch


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


def normalize(train, test, crossval):
    scaler = preprocessing.StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    crossval_scaled = scaler.transform(crossval)
    return train_scaled, test_scaled, crossval_scaled


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
