# -*- coding: utf-8 -*-
#

"""
Filename: dataset_perTrafficType.py
Date: Fri Oct 19 17:47:53 2018
Name: Arnaud Rosay
Description:
    - Read all CSV file containing labels
    - Group by traffic type (label)
    - Generate one CSV file (incl. label) per traffic type
"""

import pandas as pd


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


def index2string(index):
    """
    Convert an int to string

    Parameter
    ---------
    index: int
        index to be converted

    Return
    ------
    string: string
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
        index = -1
    return string


def get_dataframe_ofType(df, traffic_type):
    """
    Analyze traffic distribution of pandas data frame containing IDS2017 CSV
    file with labelled traffic

    Parameter
    ---------
    df: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file
    traffic_type: string
        name corresponding to traffic type

    Return
    ------
    req_df: DataFrame
        Pandas DataFrame containing only the requested traffic type

    """
    # select the right rowsget
    req_df = df.loc[df[' Label'] == traffic_type]
    # don't keep original indexes
    req_df = req_df.reset_index()
    return req_df


def main():
    # declare useful variables
    # path = "../../01_Common/05_IDS2017/TrafficLabelling/"
    path = "~/PycharmProjects/test/data/TrafficLabelling/"
    filelist = ("Monday-WorkingHours.pcap_ISCX.csv",
                "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv",
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
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
    dflist = [pd.DataFrame(),  # benign
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
    # loop over each file
    for filename in filelist:
        print("[INFO] Extracting from ", filename, "...")
        # Load one file as a data frame
        df = pd.read_csv(path + filename,
                         encoding="ISO-8859-1", low_memory=False)
        typelist = get_typelist(df)
        for type in typelist:
            dflist[string2index(type)] = dflist[string2index(type)].append(
                    get_dataframe_ofType(df, type))
    # display size
    print("[INFO] Shape of generated DataFrames:")
    for idx in range(15):
        print("shape: {} \t- {}".format(dflist[idx].shape, index2string(idx)))
    # create one CSV file per traffic type
    print("[INFO] Writing CSV files ...")
    for idx in range(15):
        dflist[idx].to_csv(path+csvlist[idx], encoding="ISO-8859-1")


if __name__ == "__main__":
    main()
