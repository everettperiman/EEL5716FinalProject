import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
#import kerastuner as kt
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters

def linear_attack(train_features, test_features, train_labels, test_labels,n_samples):
    # cut down the # of challenges to 50,000 for traditional ml algo such as svm and lr
    tr_f = train_features[:n_samples]
    tr_l = train_labels[:n_samples]
    te_f = test_features[n_samples:n_samples + int(1.2 * n_samples)]
    te_l = test_labels[n_samples:n_samples + int(1.2 * n_samples)]
    print("1. Linearly Separable Test")
    print('   Trying to attack with %d samples' %n_samples)
    print("    1.a SVM Classifier")
    lin_svc = svm.LinearSVC(C=1.0).fit(train_features, train_labels)
    y_pred = lin_svc.predict(test_features)
    acc_svm = accuracy_score(test_labels, y_pred)
    print('        Linear SVM Accuracy: %f\n' % accuracy_score(test_labels, y_pred))
    print("    1.b Logistic Regression Classifier")
    lin_lr = LogisticRegression(random_state=0).fit(train_features, train_labels)
    y_pred = lin_lr.predict(test_features)
    acc_lr = accuracy_score(test_labels, y_pred)
    print('        Logistic Regression Accuracy: %f\n' % accuracy_score(test_labels, y_pred))
    if ((acc_svm > 0.8) or (acc_lr > 0.8)):
        print('The challenges and responses are linearly separable with accuracy %f',  np.max(acc_svm,acc_lr)*100)
    elif (((acc_svm < 0.8) and (acc_svm > 0.6))  or ((acc_lr < 0.8) and (acc_lr > 0.6))):
        print("Not satisfactory to conculde linearly separabality. Maximum achieved accuracy is ", np.max(acc_svm,acc_lr))
    else :
        print(" The challenges and responses are NOT linearly separable")

def extract_sets(filename="CRPSets.xls"):
    df = pd.DataFrame(pd.read_excel(filename, names=[str(i) for i in range(65)], header=None))
    challenges = df.iloc[:, :-1].to_numpy()
    responses = df.iloc[:, -1].to_numpy()


    return challenges, responses

def split_data(x, y, train_sample_size=2000):
    train_size = train_sample_size / 12000
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, shuffle=True, random_state=False)
    return x_train, x_test, y_train, y_test

def main():
    # Import data
    challenges, responses = extract_sets()
    # Display data

    # Plot possible correlations

    # Split the data into testing and training data
    # X represents the challenges where y represents the responses
    x_train, x_test, y_train, y_test = split_data(challenges, responses, train_sample_size = 11000)

    print("Attack Time")
    # Train the model
    linear_attack(x_train, x_test, y_train, y_test, 12000)
    print("Attack Time Done")
    # Evaluate the model

if __name__ == "__main__":
    main()