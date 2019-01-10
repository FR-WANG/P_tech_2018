import numpy as np
import pandas as pd
from sklearn import preprocessing


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
