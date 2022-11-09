import numpy as np
import pandas as pd

def convert_to_feature_vectors(challenges):
    for index, challenge in enumerate(challenges):
        challenges[index] = 2*challenge - 1
    return challenges

def extract_sets(filename="CRPSets.xls"):
    if '.xls' in filename:
        df = pd.DataFrame(pd.read_excel(filename, names=[str(i) for i in range(65)], header=None))
    if '.csv' in filename:
        df = pd.DataFrame(pd.read_csv(filename, names=[str(i) for i in range(65)], header=None))
    challenges = convert_to_feature_vectors(df.iloc[:, :-1].to_numpy())
    trans_challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
    responses = convert_to_feature_vectors(df.iloc[:, -1].to_numpy()).reshape(-1,1)
    return challenges, trans_challenges, responses


def main():
    # Import data
    challenges, trans_challenge, responses = extract_sets('simulated_crpsets.csv')
    challenges = [[1, 2, 3], [4, 5, 6]]
    a = np.fliplr(challenges) # Flips the direction now in order from closest to furthest from challenge
    b = np.cumprod(a, axis=1, dtype=np.int8) # Evaluates the product 
    print(challenges[0])
    print(a[0])
    print(b[0])


if __name__ == "__main__":
    main()