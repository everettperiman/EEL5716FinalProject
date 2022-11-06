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

def squish_rows(arr):
    return_arr = []
    for row in arr:
        arr_bin_string = ''.join([str(i) for i in row])
        return_arr.append(int(arr_bin_string,2))
    print(return_arr)
    np_arr = np.array(return_arr)
    np_arr = np_arr.reshape(-1, 1)
    return MinMaxScaler(feature_range=(0, 1)).fit_transform(X=np_arr, y=None)

def split_data(x, y, train_sample_size=2000):
    train_size = train_sample_size / 12000
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, shuffle=True, random_state=False)
    return x_train, x_test, y_train, y_test

def svm_test(x_train, x_test, y_train, y_test):
    from sklearn import svm
    clf = svm.SVC(kernel='rbf')
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

def main():
    # Import data
    challenges, responses = crp_sets()
    # Display data
    print(challenges)

    # Plot possible correlations
    #plt.plot(responses, challenges, 'o')
    #plt.show()

    # Split the data into testing and training data
    # X represents the challenges where y represents the responses
    x_train, x_test, y_train, y_test = split_data(challenges, responses, train_sample_size = 10000)
    print(x_train)
    print(y_train)

    # Train the model
    #svm_test(x_train, x_test, y_train, y_test)
    mlp_class(x_train, x_test, y_train, y_test)


    # Evaluate the model




if __name__ == "__main__":
    main()