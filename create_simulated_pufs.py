import pandas as pd
import numpy as np


def convert_to_feature_vectors(challenges):
    for index, challenge in enumerate(challenges):
        challenges[index] = 2*challenge - 1
    return challenges

# This extracts the CRPs from the excel files
# Converts the challenges to their respective feature vectors
def extract_sets(filename="CRPSets.xls"):
    if '.xls' in filename:
        df = pd.DataFrame(pd.read_excel(filename, names=[str(i) for i in range(65)], header=None))
        challenges = convert_to_feature_vectors(df.iloc[:, :-1].to_numpy().astype(np.int8))
        responses = convert_to_feature_vectors(df.iloc[:, -1].to_numpy().astype(np.float64)).reshape(-1,1)
    
    if '.csv' in filename:
        df = pd.DataFrame(pd.read_csv(filename, names=[str(i) for i in range(65)], header=None))
        challenges = convert_to_feature_vectors(df.iloc[:, :-1].to_numpy())
        responses = convert_to_feature_vectors(df.iloc[:, -1].to_numpy()).reshape(-1,1)
    
    return challenges, responses



def attack_crps():
    import pypuf.simulation, pypuf.io, pypuf.attack   
    challenges, responses = extract_sets()
    crps = pypuf.io.ChallengeResponseSet(challenges, responses)
    attack = pypuf.attack.LRAttack2021(crps, seed=3, k=4, bs=1000, lr=.001, epochs=100)
    attack.fit()
    attack = pypuf.attack.MLPAttack2021(crps, seed=3, net=[2 ** 4, 2 ** 5, 2 ** 4],epochs=30, lr=.001, bs=1000, early_stop=.08)
    attack.fit()


def main():
    from pypuf.simulation import ArbiterPUF
    from pypuf.io import random_inputs
    from pypuf.io import ChallengeResponseSet
    import csv
    
    desired_number_of_pufs = 50000
    puf = ArbiterPUF(n=64, seed=1)
    crps = ChallengeResponseSet.from_simulation(puf, N=desired_number_of_pufs, seed=2)
    one_zero_crps = []
    with open('simulated_crpsets.csv','w') as csvfile:
        for crp in crps:
            challenge = crp[0]
            response = crp[1]
            if response == -1:
                response = 0
            else:
                response = 1
            new_challenge = []
            for digit in challenge:
                if digit == -1:
                    new_challenge.append(0)
                else:
                    new_challenge.append(1)
            new_challenge.append(response)
            csvfile.write(','.join([str(i) for i in new_challenge]) + '\n')
        


if __name__ == "__main__":
    attack_crps()
    #main()