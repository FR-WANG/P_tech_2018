# -*- coding: utf-8 -*-
#

"""
Filename: dataset_extraction.py
Date: Mon Oct 22 14:05:13 2018
Name: Arnaud Rosay
Description:
    -  Extract samples of each traffic type to generate training set,
    cross-validation set and test set

"""

import pandas as pd
import numpy as np


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
    Convert a string to int so that it can be used as index in an list
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


def index2string(index):
    """
    Convert an int to string
    Parameter
    ---------
    index: int
        index to be converted
    Return
    ------
    string: str
        string corresponding to the string
    """
    if index == 0:
        string = 'BENIGN'
    elif index == 1:
        string = 'FTP-Patator'
    elif index == 2:
        string = 'SSH-Patator'
    elif index == 3:
        string = 'DoS Hulk'
    elif index == 4:
        string = 'DoS GoldenEye'
    elif index == 5:
        string = 'DoS slowloris'
    elif index == 6:
        string = 'DoS Slowhttptest'
    elif index == 7:
        string = 'Heartbleed'
    elif index == 8:
        string = 'Web Attack Brute Force'
    elif index == 9:
        string = 'Web Attack XSS'
    elif index == 10:
        string = 'Web Attack Sql Injection'
    elif index == 11:
        string = 'Infiltration'
    elif index == 12:
        string = 'Bot'
    elif index == 13:
        string = 'PortScan'
    elif index == 14:
        string = 'DDoS'
    else:
        print("[ERROR] Cannot convert {}".format(index))
        string = 'Error'
    return string


def random_permutation(df_list):
    """
    Run permutations in the dataset
    Parameters
    ---------
    df_list: list
        list of pandas DataFrames, each DataFrames containing one traffic type
    Return
    ------
    reordered_df_list: list
        Resulting list of pandas DataFrames
    """
    df_list_size = 15
    reordered_df_list = df_list
    for idx in range(df_list_size):
        # Shuffle rows
        reordered_df_list[idx] = df_list[idx].sample(frac=1, replace=False)
    return reordered_df_list


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
    # write stats about traffic in an list
    idx = 0
    for lbl in labels_list:
        stats[string2index(lbl), 0] = labels[idx]
        idx = idx + 1
    # create a data frame from the list
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


def split_dataset(df_list, randomize=False, benign_clipping=True,
                  training_percentage=50,
                  crossval_percentage=25,
                  test_percentage=25):
    """
    Split a dataset provided as an list of pandas DataFrames, each DataFrames
    containing one traffic type

    Parameter
    ---------
    df_list: list
        list of pandas DataFrames
    randomize: boolean
        When True, data lines are permuted randomly before splitting datasets
        (default = False)
    benign_clipping: boolean
        When True, amount of traffic is set to represent 50% of overall
        traffic
    training_percentage: int
        Value between 0 and 100 corresponding to the percentage of dataset
        used to create the training set (default = 60% of dataset)
    dev_percentage: int
        Value between 0 and 100 corresponding to the percentage of dataset
        Note that values can be a float as long as percentage of dataset
        is an integer
        used to create the training set (default = 20% of dataset)
    test_percentage: int
        Value between 0 and 100 corresponding to the percentage of dataset
        used to create the training set (default = 20% of dataset)

    Return
    ------
    train_set: DataFrame
        Pandas DataFrame used as training set
    cv_set: DataFrame
        Pandas DataFrame used as cross validation set
    test_set: DataFrame
        Pandas DataFrame used as test set

    """
    print("[INFO] Splitting dataset")
    # Check percentage values
    if training_percentage + crossval_percentage + test_percentage != 100:
        print("[ERROR] Sum of percentages != 100")
        exit(-1)
    # Randomize dataset if requested
    if randomize is True:
        df_list = random_permutation(df_list)
    # Declare DataFrame to be returned
    train_set = pd.DataFrame()
    cv_set = pd.DataFrame()
    test_set = pd.DataFrame()
    # Select subset of each dataset except Benign traffic
    df_list_size = 15
    for idx in range(1, df_list_size):
        n_rows = df_list[idx].shape[0]
        n_training = int(n_rows * training_percentage / 100)
        n_crossval = int(n_rows * crossval_percentage / 100)
        n_test = int(n_rows * test_percentage / 100)
        if index2string(idx) == 'BENIGN' and benign_clipping is True:
            # Limit BENIGN traffic so that amount of BENIGN examples in attack
            # files is the same as amount of attack examples
            clipping_value = n_crossval
            training_end = clipping_value
        else:
            training_end = n_training
        crossval_end = training_end + n_crossval
        test_end = crossval_end + n_test
        train_set = train_set.append(df_list[idx][:training_end])
        cv_set = cv_set.append(df_list[idx][training_end:crossval_end])
        test_set = test_set.append(df_list[idx][crossval_end:test_end])
    # Handle specific case of Benign traffic
    n_train_attacks = train_set.shape[0]
    training_end = n_train_attacks
    n_cv_attacks = cv_set.shape[0]
    crossval_end = training_end + n_cv_attacks
    n_test_attacks = test_set.shape[0]
    test_end = crossval_end + n_test_attacks
    train_set = train_set.append(df_list[0][:training_end])
    cv_set = cv_set.append(df_list[0][training_end:crossval_end])
    test_set = test_set.append(df_list[0][crossval_end:test_end])
    # Shuffle datasets
    train_set = train_set.sample(frac=1)
    cv_set = cv_set.sample(frac=1)
    test_set = test_set.sample(frac=1)
    # Display size of each DataFrame
    print("Training set shape: {}".format(train_set.shape))
    print("Cross-val set shape: {}".format(cv_set.shape))
    print("Test set shape: {}".format(test_set.shape))
    # return resulting DataFrames
    return train_set, cv_set, test_set


def detect_drop_outliers(df):
    """
    Detect and drop NaN rows of a DataFrame

    Parameters
    ----------
    df: DataFrame
        pandas DataFrame containing data

    Returns
    -------
    clean_df: DataFrame
        pandas DataFrame without rows containing NaN

    """


    # flow equals to 0 count

    nb_rows = len(df['Flow Bytes/s'].index)
    print(nb_rows)
    for i in range(0,nb_rows-1):
        #print(i,df[' Flow Packets/s'][i])
        if (df[' Flow Packets/s'][i] != 0):
            if(df['Flow Bytes/s'][i] == 0):
                print(i,'sb',df[' Flow Packets/s'][i],df['Flow Bytes/s'][i])

    df.replace(['+Infinity', '-Infinity', 'Infinity'], np.nan, inplace=True)
    null_columns = df.columns[df.isna().any()]
    nan_cnt = df[null_columns].isnull().sum()
    if nan_cnt.empty is False:
        print("NaN detected and dropped")
        print(nan_cnt)
        clean_df = df.dropna(axis=0)
        print("Prev shape: {} - New shape: {}".format(df.shape, clean_df.shape))
    else:
        clean_df = df
    # special case of 'Flow Bytes/s' and ' Flow Packets/s' where data is stored
    # as str
    lbl_list = df.columns.values
    lbl_idx, = np.where(lbl_list=='Flow Bytes/s')
    clean_df['Flow Bytes/s'] = np.array(clean_df.iloc[:, lbl_idx]).astype(float)
    lbl_idx, = np.where(lbl_list == ' Flow Packets/s')
    clean_df[' Flow Packets/s'] = np.array(clean_df.iloc[:, lbl_idx]).\
        astype(float)
    # negative values in positive fields[lbl]
    for lbl in lbl_list:
        if lbl != ' Label':
            neg_cnt = clean_df[lbl].apply(lambda x: x < 0).value_counts()
            if neg_cnt.shape[0] > 1:
                neg_cnt = neg_cnt[1]
            else:
                neg_cnt = 0
            if neg_cnt != 0:
                print("'{}' - Neg values: {}".format(lbl, neg_cnt))


    # return

    return clean_df


def analyze_dist(df):
    """
    Analyze distribution of the provided DataFrame

    Parameters
    ----------
    df: DataFrame
        pandas DataFrame containing data

    Returns
    -------
    None

    """
    lbl_list = df.columns.values
    for lbl in lbl_list:
        if lbl != ' Label':
            print("***********  {}  ***********".format(lbl))
            try:
                min_val = np.amin(df[lbl])
                max_val = np.amax(df[lbl])
                mean_val = np.mean(df[lbl])
                std_val = np.std(df[lbl])
                hist, _ = np.histogram(df[lbl], bins=10)

                print("min: {:.1f} - max: {:.1f} - mean: {:.1f} - std: {:.1f}".
                      format(min_val, max_val, mean_val, std_val))
                print("Hist: {}".format(hist))
                #print("Neg value count: {}".format(neg_cnt))
            except TypeError:
                print("[ERROR] Cannot analyze due to TypeError")


def main():
    # declare useful variables
    # path = "../../01_Common/05_IDS2017/TrafficLabelling/"
    path = "~/PycharmProjects/test/data/TrafficLabelling/"
    #path = "D://ml/ai4nids/polytech/01_Common/05_IDS2017/TrafficLabelling/"
    csvlist = ["BENIGN.csv",
               "FTP-Patator.csv",
               "SSH-Patator.csv",
               "DoS Hulk.csv",
               "DoS GoldenEye.csv",
               "DoS slowloris.csv",
               "DoS Slowhttptest.csv",
               "Heartbleed.csv",
               "Web Attack Brute Force.csv",
               "Web Attack XSS.csv",
               "Web Attack Sql Injection.csv",
               "Infiltration.csv",
               "Bot.csv",
               "PortScan.csv",
               "DDoS.csv"]
    df_list = [pd.DataFrame(),  # benign
               pd.DataFrame(),  # ftp_patator
               pd.DataFrame(),  # ssh_patator
               pd.DataFrame(),  # dos_hulk
               pd.DataFrame(),  # dos_goldeneye
               pd.DataFrame(),  # dos_slowloris
               pd.DataFrame(),  # dos_slowhttptest
               pd.DataFrame(),  # heartbleed
               pd.DataFrame(),  # webattack_bruteforce
               pd.DataFrame(),  # webattack_xss
               pd.DataFrame(),  # webattack_sqlinjection
               pd.DataFrame(),  # infiltration
               pd.DataFrame(),  # bot
               pd.DataFrame(),  # portscan
               pd.DataFrame()]  # ddos
    # loop over each file to load DataFrames
    idx = 0
    for filename in csvlist:
        print("[INFO] Reading ", filename, "...")
        # Load one file as a data frame
        df_list[idx] = pd.read_csv(path + filename,
                                   encoding="ISO-8859-1", low_memory=False)
        # Remove previous index column
        df_list[idx] = df_list[idx].drop(columns='Unnamed: 0', axis=1)
        # Remove useless columns
        drop_list = ['index', 'Flow ID', ' Source IP', ' Destination IP', ' Source Port',
                     ' Destination Port', ' Timestamp']
        for lbl in drop_list:
            df_list[idx] = df_list[idx].drop(columns=lbl, axis=1)
        # detect outliers
        df_list[idx] = detect_drop_outliers(df_list[idx])
        idx = idx + 1
    # generate 3 subsets for training, cross validation and test
    print("[INFO] Generating training, cross-val and test DataFrames ...")
    (df_train, df_cv, df_test) = split_dataset(df_list,
                                               randomize=True,
                                               benign_clipping=True,
                                               training_percentage=50,
                                               crossval_percentage=25,
                                               test_percentage=25)
    # write in CSV file
    print("[INFO] Writing CSV files ...")
    df_train.to_csv(path + "train_set.csv", encoding="ISO-8859-1")
    df_cv.to_csv(path + "crossval_set.csv", encoding="ISO-8859-1")
    df_test.to_csv(path + "test_set.csv", encoding="ISO-8859-1")
    # analyze distribution of features for each set
    analyze_dist(df_train)
    # analyze_dist(df_cv)
    # analyze_dist(df_test)


if __name__ == "__main__":
    main()
