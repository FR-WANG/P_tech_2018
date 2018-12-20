# -*- coding: utf-8 -*-
#

"""
Filename: dataset_exploration.py
Date: Tue Oct 16 11:26:11 2018
Name: Arnaud Rosay
Description:
    -  Program to investigate dataset content
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_typelist(df):

    """
    Extract traffic type from a pandas data frame containing IDS2017 CSV
    file with labelled traffic

    Parameter
    ---------
    df: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file

    Return
    ------
    traffic_type_list: list
        List of traffic types contained in the DataFrame

    """
    traffic_type_list = df[' Label'].value_counts().index.tolist()

    return traffic_type_list


def string2index(string):

    """
    Convert a string to int so that it can be used as index in an array

    Parameter
    ---------
    string: string
        string to be converted

    Return
    ------
    index: int
        index corresponding to the string

    """
    if string == 'BENIGN':
        index = 0
    elif string == 'FTP-Patator':
        index = 1
    elif string == 'SSH-Patator':
        index = 2
    elif string == 'DoS Hulk':
        index = 3
    elif string == 'DoS GoldenEye':
        index = 4
    elif string == 'DoS slowloris':
        index = 5
    elif string == 'DoS Slowhttptest':
        index = 6
    elif string == 'Heartbleed':
        index = 7
    elif string == 'Web Attack \x96 Brute Force':
        index = 8
    elif string == 'Web Attack \x96 XSS':
        index = 9
    elif string == 'Web Attack \x96 Sql Injection':
        index = 10
    elif string == 'Infiltration':
        index = 11
    elif string == 'Bot':
        index = 12
    elif string == 'PortScan':
        index = 13
    elif string == 'DDoS':
        index = 14
    else:
        print("[ERROR] Cannot convert ", string)
        index = -1
    return index


def get_traffic(dataframe):
    """
    Analyze traffic of pandas data frame containing IDS2017 CSV file with
    labelled traffic

    Parameter
    ---------
    dataframe: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file

    Return
    ------
    df_stats: DataFrame
        Returns a pandas data frame of one column containing amount of lines
        for each type of traffic

    """
    stats = np.zeros(15)
    stats = stats.reshape(15, 1)
    # check that all samples have been labeled
    n_samples = dataframe.shape[0]
    labels = dataframe[' Label'].value_counts()
    labels_list = get_typelist(dataframe)
    n_labels = labels.sum()
    if n_labels != n_samples:
        print("\t[INFO] missing labels: {}".format(n_samples - n_labels))
    else:
        print("\t[INFO] no missing labels")
    # write stats about traffic in an array
    idx = 0
    for lbl in labels_list:
        stats[string2index(lbl), 0] = labels[idx]
        idx = idx + 1
    # create a data frame from the array
    df_stats = pd.DataFrame(stats,
                            index=['BENIGN',
                                   'FTP-Patator',
                                   'SSH-Patator',
                                   'DoS Hulk',
                                   'DoS GoldenEye',
                                   'DoS Slowloris',
                                   'DoS Slowhttptest',
                                   'Heartbleed',
                                   'WebAttack Brute Force',
                                   'WebAttack XSS',
                                   'WebAttack SQL Injection',
                                   'Infiltration',
                                   'Bot',
                                   'PortScan',
                                   'DDoS'])
    print("\t[INFO] Analysis done")
    return df_stats


def get_traffic_distribution(dataframe):
    """
    Analyze traffic distribution of pandas data frame containing IDS2017 CSV
    file with labelled traffic

    Parameter
    ---------
    dataframe: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file

    Return
    ------
    df_stats: DataFrame
        Returns a pandas data frame of one column containing amount of lines
        for each type of traffic

    """
    df = pd.DataFrame(dataframe[' Label'])
    df.columns = ['label']
    df['change'] = (df['label'] != df['label'].shift())
    df['change'] = df['change'].astype(int)
    index = df['change'].nonzero()
    index_t = np.transpose(index)
    idx = pd.DataFrame(index_t, columns=['index'])
    diff = pd.DataFrame(abs(idx.diff(-1)))
    diff.columns = ['seq_len']
    diff['seq_len'][diff['seq_len'].shape[
        0] - 1] = int(df.shape[0] - index_t[-1])
    diff['seq_len'] = diff['seq_len'].astype(int)
    labels = np.array(df['label'].iloc[index])
    labels = np.reshape(labels, (np.shape(labels)[0], 1))
    lbl = pd.DataFrame(labels, columns=['label'])
    stats = pd.concat([idx, lbl, diff], axis=1)
    for traffic_type in stats['label'].value_counts().index.tolist():
        df_perTraffic = stats.loc[stats['label'] == traffic_type]
        first = np.array(df_perTraffic.head(1)['index'])[0]
        last = np.array(df_perTraffic.tail(1)['index'])[0] + \
            np.array(df_perTraffic.tail(1)['seq_len'])[0]
        print("\t", traffic_type,
              "first frame: {}".format(first))
        print("\t", traffic_type,
              "last frame: {}".format(last))
        print("\t", traffic_type,
              "number of seq: {}".format(df_perTraffic.shape[0]))
        print("\t", traffic_type,
              "min seq_len: {}".format(
                  pd.DataFrame.min(df_perTraffic['seq_len'])))
        print("\t", traffic_type,
              "max seq_len: {}".format(
                  pd.DataFrame.max(df_perTraffic['seq_len'])))
        print("\t", traffic_type,
              "mean seq_len: {}".format(
                  pd.DataFrame.mean(df_perTraffic['seq_len'])))
        print("\t", traffic_type,
              "std seq_len: {}".format(
                  pd.DataFrame.std(df_perTraffic['seq_len'])))
        if traffic_type != 'BENIGN':
            if pd.DataFrame.max(df_perTraffic['seq_len']) < 30:
                step = 1
            else:
                step = pd.DataFrame.max(df_perTraffic['seq_len']) / 30
            bin_range = np.arange(0,
                                  pd.DataFrame.max(
                                      df_perTraffic['seq_len']) + step,
                                  step)
            out, bins = pd.cut(df_perTraffic['seq_len'], bins=bin_range,
                               include_lowest=True, right=False, retbins=True)
            out.value_counts(sort=False).plot.bar()
            plt.show()
    return stats


def main():
    # declare useful variables
    path = "~/PycharmProjects/test/data/TrafficLabelling/"
    filelist = ("Monday-WorkingHours.pcap_ISCX.csv",
                "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv",
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    global_traffic = pd.DataFrame()
    file_traffic = pd.DataFrame()
    # loop over each file
    idx = 0
    for filename in filelist:
        # Load one file as a data frame
        print("[INFO] Analysing ", filename, "...")
        df = pd.read_csv(path + filename,
                         encoding="ISO-8859-1", low_memory=False)
        file_traffic = get_traffic(df)
        global_traffic = pd.concat([global_traffic, file_traffic], axis=1)
        get_traffic_distribution(df)
        idx = idx + 1
    global_traffic.columns = ["Mon-Normal",
                              "Tue-BruteForce",
                              "Wed-doS",
                              "ThuAM-WebAttacks",
                              "ThurPM-Infiltration",
                              "FriAM-Bot.csv",
                              "FriPM-PortScan",
                              "FriPM-DDos"]
    global_traffic['Total'] = global_traffic.sum(axis=1)
    n_samples = global_traffic['Total'].sum(axis=0)
    global_traffic['%'] = \
        global_traffic['Total'].apply(lambda x: 100 * x / n_samples)

    # store data frames in excel files
    writer = pd.ExcelWriter('IDS2017-dataset-analysis.xlsx')
    global_traffic.to_excel(writer, 'Traffic')
    writer.save()


if __name__ == "__main__":
    main()
