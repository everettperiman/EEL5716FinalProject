from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def crp_sets(filename="CRPSets.xls"):
    df = pd.DataFrame(pd.read_excel(filename, names=[str(i) for i in range(65)], header=None))
    #challenges = squish_rows(df.iloc[:, :-1].to_numpy()) # Compress and scale each challenge
    challenges = df.iloc[:, :-1].to_numpy()
    responses = df.iloc[:, -1].to_numpy()
    return challenges, responses



def main():
    # Import data
    challenges, responses = crp_sets()
    # Display data
    print(challenges)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=64)
    pca.fit(challenges)
    total = 0
    index = 0
    x = list(pca.explained_variance_ratio_)
    x.sort(reverse=True)
    for i in x:
        total = total + i
        index = index + 1
        if total > .9:
            break
    print(index)
if __name__ == "__main__":
    main()