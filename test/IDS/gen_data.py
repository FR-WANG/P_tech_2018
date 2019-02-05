
import numpy as np
import pandas as pd

from sklearn import preprocessing
import tensorflow as tf


def shuffle(data, labels, batch_size):
    """
    Function to shuffle the data and labels

    Parameter
    ---------

    data : DataFrame
        gives the data to shuffle
    labels : DataFrame
        gives the labels to shuffle
    batch_size : int
        number of data in a batch

    Return
    ------

    data : DataFrame
        gives the data after shuffling
    labels : DataFrame
        gives the labels after shuffling
    batch_number : int
        number of batchs in a epoch
    """
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    batch_number = int(len(data) / batch_size)

    return data, labels, batch_number


def batch(data, label, num, batch_size):
    """
    Function to divide the data and labels into batches

    Parameter
    ---------

    data : DataFrame
        gives the data to be divided
    label : DataFrame
        gives the labels to be divided
    batch_size : int
        number of data in a batch

    Return
    ------

    data_batch : DataFrame
        gives the data of one batch
    label_batch : DataFrame
        gives the labels of one batch
    """
    data_batch = data[(num * batch_size): ((num + 1) * batch_size) - 1]
    label_batch = label[(num * batch_size): ((num + 1) * batch_size) - 1]
    return data_batch, label_batch


# load data
def loaddata():
    """
    Function to read csv file

    Parameter
    ---------


    Return
    ------


    """
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
    """
    Function to change labels from text to numbers

    Parameter
    ---------

    train, test, crossval : DataFrame
        the labels need to be changed
    Return
    ------

    train, test, crossval : DataFrame
        the labels changed
    """
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


def normalize(train, test, crossval):
    """
    Function to imply the normalization

    Parameter
    ---------

    train, test, crossval : DataFrame
        the data need to be normalized
    Return
    ------

    train_scaled, test_scaled, crossval_scaled : DataFrame
        the data after normalized
    """
    scaler = preprocessing.StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    crossval_scaled = scaler.transform(crossval)
    return train_scaled, test_scaled, crossval_scaled


def one_hot_coding(y, num_classes=None):
    """
    Function to imply the one_hot_coding

    Parameter
    ---------

    y : DataFrame
        the data need to be coded with one_hot
    Return
    ------

    categorical: DataFrame
        the data after coded with one_hot
    """
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
