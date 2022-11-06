from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extract_sets(filename="CRPSets.xls"):
    df = pd.DataFrame(pd.read_excel(filename, names=[str(i) for i in range(65)], header=None))
    challenges = df.iloc[:, :-1].to_numpy()
    responses = df.iloc[:, -1].to_numpy()
    return challenges, responses

def split_data(x, y, train_sample_size=2000):
    train_size = train_sample_size / 12000
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, shuffle=True, random_state=False)
    return x_train, x_test, y_train, y_test

def svm_test(x_train, x_test, y_train, y_test):
    from sklearn import svm
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(y_pred)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.show()
    print(metrics.accuracy_score(y_test, y_pred))

def mlp_class(x_train, x_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1, max_iter=20000000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    #disp.plot()
    plt.show()
    print(metrics.accuracy_score(y_test, y_pred))
    return 1

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
    challenges, responses = extract_sets()
    # Display data
    print(challenges)
    challenges = condition_challenges(challenges)
    # Plot possible correlations

    plt.plot(challenges, responses, 'o')
    plt.show()
    quit()
    # Split the data into testing and training data
    # X represents the challenges where y represents the responses
    x_train, x_test, y_train, y_test = split_data(challenges, responses, train_sample_size = 11000)
    print(x_train)
    print(y_train)

    # Train the model
    svm_test(x_train, x_test, y_train, y_test)
    #mlp_class(x_train, x_test, y_train, y_test)


    # Evaluate the model




if __name__ == "__main__":
    main()