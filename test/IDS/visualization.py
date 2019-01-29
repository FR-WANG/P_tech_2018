import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(confusion_matrix):

    df_cm = pd.DataFrame(
        confusion_matrix, index=[
            i for i in "ABCDEFGHIJKLMNO"], columns=[
            i for i in "ABCDEFGHIJKLMNO"])

    plt.figure(figsize=(10, 7))

    sn.heatmap(df_cm, annot=True)
    plt.show()

