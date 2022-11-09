from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def into_features_vect(chall):
    phi = []
    for i in range(1,len(chall)):
        s = sum(chall[i:])
        if s % 2 == 0:
            phi.append(1)
        else:
            phi.append(-1)
    phi.append(1)
    return phi

def crp_sets(filename="CRPSets.xls"):
    df = pd.DataFrame(pd.read_excel(filename, names=[str(i) for i in range(65)], header=None))
    #challenges = squish_rows(df.iloc[:, :-1].to_numpy()) # Compress and scale each challenge
    challenges = df.iloc[:, :-1].to_numpy()
    responses = df.iloc[:, -1].to_numpy()
    return challenges, responses

def condition_challenges(challenges):
    print(challenges.shape)
    conditioned_challenges = np.empty((12000,1))
    for index, challenge in enumerate(challenges):
        weight = 0
        for bit in challenge:
            if bit == 1:
                weight = weight + 1
        conditioned_challenges[index] = weight
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #conditioned_challenges = scaler.fit_transform(conditioned_challenges)
    return conditioned_challenges


def main():
    # Import data
    challenges, responses = crp_sets()
    from pypuf.simulation import ArbiterPUF
    from pypuf.io import random_inputs
    puf = ArbiterPUF(n=64, seed=1)
    print(puf.eval(random_inputs(n=64, N=3, seed=2)))

    #print(into_features_vect(challenges[0].tolist()))
    quit()
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