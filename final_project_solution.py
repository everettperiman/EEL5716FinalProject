import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# Additional sourcesd
# https://archive.ics.uci.edu/ml/datasets/Physical+Unclonable+Functions


TESTING = True
BEST_PARAM_FILE = 'params.txt'
BEST_PARAM_FOLDER = 'Params\\'


"""
ADMIN TOOLS
"""
# Create the graphs folder
def setup_folders():
    if not os.path.isdir('graphs'):
        os.mkdir('graphs')



"""
FEATURE IMPORT AND EXTRACTION
"""
# Convert the binary values [0,1] to ML appropriate values [-1,1]
# This improves the ability for the models to train against the data
def convert_to_feature_vectors(challenges):
    for index, challenge in enumerate(challenges):
        challenges[index] = 2*challenge - 1
    return challenges

def feed_forward_model_features(challenges):
    # Basis for transform function "Towards Fast and Accurate Machine Learning Attacks of Feed-Forward Arbiter PUFs"
    # PyPUF Python Library, mlp2021 attack

    # It is based off of the feature extraction used to model a feed forward style of PUF
    # Since an Arbiter PUF feeds forward the result of each line we can apply this model here
    # This line first reverses each challenge and then applies the cumulative product against each succesive bit
    # This models the downstream effects that each bit has had, thereby creating a set of features from the challenge vectors
    return np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)

def extract_linear_features(challenges):
    return np.linalg.pinv(challenges)

# This extracts the CRPs from the excel files
# Converts the challenges to their respective feature vectors
def extract_sets(filename="CRPSets.xls", transform=False):
    
    # Use different extraction techniques for the appropriate file type
    if '.xls' in filename:
        df = pd.DataFrame(pd.read_excel(filename, names=[str(i) for i in range(65)], header=None))
    if '.csv' in filename:
        df = pd.DataFrame(pd.read_csv(filename, names=[str(i) for i in range(65)], header=None))
    
    # This is used to convert the responses to feature vectors as well
    responses = convert_to_feature_vectors(df.iloc[:, -1].to_numpy()).reshape(-1,1)
    
    # Extract the challenges from the challenge bits to the featured challenges
    # This allows the PUF to be modeled by the implemented models later on
    raw_challenges = df.iloc[:, :-1].to_numpy()
    feature_vectors = convert_to_feature_vectors(raw_challenges)
    challenges = feed_forward_model_features(feature_vectors)
    
    return challenges, responses

"""
FEATURE EVALUATION
"""
# Used to search the challenges for potential weak spots
# Helps identify if a top or bottom path on a challenge bit
# is overly biased 
def characterize_data(challenges, responses):
    TOP, BOTTOM = 1, 0
    nodes = [{'top':0, 'bottom':0, 'percentage':0} for i in range(len(challenges[0]))]
    for challenge in challenges:
        for index, node in enumerate(challenge):
            if node == TOP:
                nodes[index]['top'] = nodes[index]['top'] + 1
            else:
                nodes[index]['bottom'] = nodes[index]['bottom'] + 1
    for node in nodes:
        node['percentage'] = node['top'] / len(challenges)
    return_list = [node['percentage'] for node in nodes]
    return_list.sort()
    return return_list

# Demonstrates the consistency of the different models evaluated
# The simulated model demonstrates that it is created in a normal distribution
# The real model shows that the hardware is more secure than the simulated model
# The Normal distribution is also plotted for reference
def demonstrate_model_consistency():
    # Extract the crp information from the simulated and the real crpsets
    # The simulated crp information was generated using the PyPUF Library and contains 50,000 crps
    simulated_challenges, simulated_responses = extract_sets('simulated_crpsets.csv', transform=True)
    real_challenges, real_responses = extract_sets('CRPSets.xls', transform=True)
    # https://www.geeksforgeeks.org/how-to-plot-a-normal-distribution-with-matplotlib-in-python/
    
    from scipy.stats import norm
    import statistics
    
    # Create the collections of values for each model
    ref_x = np.arange(.49, .51, 0.001)
    real_x = characterize_data(real_challenges, real_responses)
    sim_x = characterize_data(simulated_challenges, simulated_responses)

    # Calculate the mean and standard deviation for each model
    mean = statistics.mean(ref_x)
    sd = statistics.stdev(ref_x)
    real_mean = statistics.mean(real_x)
    real_sd = statistics.stdev(real_x)
    sim_mean = statistics.mean(sim_x)
    sim_sd = statistics.stdev(sim_x)
    
    # Plot the distributions of each model
    plt.plot(ref_x, norm.pdf(ref_x, mean, sd), label="Normal Dist.")
    plt.plot(real_x, norm.pdf(real_x, real_mean, real_sd), label="Real Dist.")
    plt.plot(sim_x, norm.pdf(real_x, sim_mean, sim_sd), label="Sim Dist.")
    plt.xlabel("Top Path preference")
    plt.ylabel("Challenge Count")
    plt.title("Distibution's of Simulated and Real CRP's")
    plt.legend(loc='upper left')
    plt.savefig('graphs\model_consistency.png')
    plt.clf()


"""
MACHINE LEARNING 
"""
def convert_continous_to_binary(predictions):
    target = (min(predictions) + max(predictions)) / 2
    for index, prediction in enumerate(predictions):
        if prediction <= target:
            predictions[index] = 0
        else:
            predictions[index] = 1
       
    return predictions

# Takes in the challenges(x) and responses(y) and converts them into a shuffled training set
# The size of the traning data is determined by the train_sample_size parameter
def split_data(x, y, train_sample_size=2000):
    train_size = train_sample_size / len(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, shuffle=True, random_state=False)
    return x_train, x_test, y_train, y_test

# Machine Learning Models

def svc(x_train, x_test, y_train, y_test, grid_search=False, print_matrix=False):
    # Use the grid search technique if desired
    if grid_search:
        svc = SVC()
        parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid',), 
                        'C':(1, 10, 100, 1000), 
                        'gamma':('auto',)}

        clf = GridSearchCV(svc, parameters, n_jobs=-1)  

    # Use best results from previous grid searchs as default values
    else:
        #{'C': 100, 'gamma': 'auto', 'kernel': 'linear'}
        clf = SVC(C=100,gamma='auto',kernel='linear')

    # Fit and predict
    clf.fit(x_train, y_train.ravel())
    y_pred = clf.predict(x_test)

    # Print model accuracy and confusion matrix
    if print_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test.ravel(), y_pred))
        disp.plot()
        plt.title("Confusion Matrix SVM @ {} Samples".format(len(x_train)))
        plt.savefig('graphs\svm_{}.png'.format(len(x_train)))
        plt.clf()

    # Print details about the grid search
    if grid_search and TESTING:
        f = open(BEST_PARAM_FOLDER + "svc_" + BEST_PARAM_FILE, "a")
        f.write(str(clf.best_params_) + "\n")
        f.close()
        svr_details = pd.DataFrame.from_dict(clf.cv_results_)
        with pd.option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
            print(svr_details)
    return accuracy_score(y_test.ravel(), y_pred)

def lgr(x_train, x_test, y_train, y_test, grid_search=False, print_matrix=False):
    
    # Use the grid search technique if desired
    if grid_search:
        lgr = LogisticRegression()
        parameters = {'penalty':('l1', 'l2', 'elasticnet', 'none',), 
                        'C':(1, 10, 100, 1000)}

        clf = GridSearchCV(lgr, parameters, n_jobs=-1)  

    # Use best results from previous grid searchs as default values
    else:
        #{'C': 100, 'penalty': 'l2'}
        clf = LogisticRegression(C=100,penalty='l2')

    # Fit and predict
    clf.fit(x_train, y_train.ravel())
    y_pred = clf.predict(x_test)

    # Print model accuracy and confusion matrix
    if print_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test.ravel(), y_pred))
        disp.plot()
        plt.title("Confusion Matrix LGR @ {} Samples".format(len(x_train)))
        plt.savefig('graphs\lgr_{}.png'.format(len(x_train)))
        plt.clf()


    # Print details about the grid search
    if grid_search and TESTING:
        f = open(BEST_PARAM_FOLDER + "lgr_" + BEST_PARAM_FILE, "a")
        f.write(str(clf.best_params_) + "\n")
        f.close()
        lgr_details = pd.DataFrame.from_dict(clf.cv_results_)
        with pd.option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
            print(lgr_details)
    return accuracy_score(y_test.ravel(), y_pred)

# Machine Model Evaluations
def evaluate_ml_models(challenges, responses):
    # Record model performance over training data sizes
    scores_svc = []
    scores_lgr = []

    # Test the model accuracy over several different training data sizes
    training_samples = [2000, 6000, 10000]
    for i in training_samples:

        # Split CRP data into training and test data
        x_train, x_test, y_train, y_test = split_data(challenges, responses, train_sample_size = i)
        print(len(x_train))

        # Train and measure the fit of each model
        # Determine whether or not we want to find the best default values
        training_yes_no = False
        print_matrix = True
        scores_svc.append(svc(x_train, x_test, y_train, y_test, training_yes_no, print_matrix))
        scores_lgr.append(lgr(x_train, x_test, y_train, y_test, training_yes_no, print_matrix))

    if scores_svc:
        plt.plot(training_samples, scores_svc, label="SVM")
    if scores_lgr:
        plt.plot(training_samples, scores_lgr, label="LGR")

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc='center')
    plt.title("Training Scores @ 2000,6000,10000")
    plt.savefig('graphs\TrainingScores.png')
    plt.clf()

# Machine Model Evaluations
def find_eighty_percent(challenges, responses):
    # Record model performance over training data sizes
    scores_svc = []
    scores_lgr = []

    # Test the model accuracy over several different training data sizes
    # Start at 20% training data and decrease by .1%
    high = int(len(challenges)*.1)
    low = int(len(challenges)*.0001)
    step = int(len(challenges) * .001)
    training_samples = list(range(low, high, step))
    training_samples.reverse()

    for i in training_samples:

        # Split CRP data into training and test data
        x_train, x_test, y_train, y_test = split_data(challenges, responses, train_sample_size = i)
        print(len(x_train))

        # Train and measure the fit of each model
        # Determine whether or not we want to find the best default values
        training_yes_no = False
        
        if (len(scores_svc) == 0) or (min(scores_svc) > .80):
            scores_svc.append(svc(x_train, x_test, y_train, y_test, training_yes_no))
        if (len(scores_lgr) == 0) or min(scores_lgr) > .80:
            scores_lgr.append(lgr(x_train, x_test, y_train, y_test, training_yes_no))
    
    if scores_svc:
        plt.plot(training_samples[:len(scores_svc)], scores_svc, label="SVM")
    if scores_lgr:
        plt.plot(training_samples[:len(scores_svc)], scores_lgr, label="LGR")

    scores = min([min(training_samples[:len(scores_svc)]), min(training_samples[:len(scores_svc)])])
    plt.hlines(y=.8, 
                xmin=scores, 
                xmax=max(training_samples),
                colors="red",
                linestyles='dashed')
    
    plt.xlim=(0)
    plt.xticks(list(range(0, max(training_samples), 100)), rotation = 45)
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc='center')
    plt.title("80% Accuracy Score")
    plt.savefig('graphs\EightyPercentScores.png')
    plt.clf()

# Machine Model Evaluations
def detailed_model_scores(challenges, responses):
    # Record model performance over training data sizes
    scores_svc = []
    scores_lgr = []

    # Test the model accuracy over several different training data sizes
    high = int(len(challenges)*.95)
    low = int(len(challenges)*.008)
    step = 50
    training_samples = list(range(low, high, step))
    for i in training_samples:

        # Split CRP data into training and test data
        x_train, x_test, y_train, y_test = split_data(challenges, responses, train_sample_size = i)
        print(len(x_train))

        # Train and measure the fit of each model
        # Determine whether or not we want to find the best default values
        training_yes_no = False
        print_matrix = True
        scores_svc.append(svc(x_train, x_test, y_train, y_test, training_yes_no))
        scores_lgr.append(lgr(x_train, x_test, y_train, y_test, training_yes_no))


    high_perf_index = []
    for index, i in enumerate(scores_svc):
        if i > .9:
            high_perf_index.append(index)
    for index, i in enumerate(scores_lgr):
        if i > .9:
            high_perf_index.append(index)
   
    if scores_svc:
        plt.plot(training_samples[min(high_perf_index):], scores_svc[min(high_perf_index):], label="SVM")
    if scores_lgr:
        plt.plot(training_samples[min(high_perf_index):], scores_lgr[min(high_perf_index):], label="LGR")

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy Score")
    plt.title("Performance above 90% Accuracy")
    plt.legend(loc='center')
    plt.savefig('graphs\HighPerfTrainingScores.png')
    plt.clf()



def main():
    # Import data
    demonstrate_model_consistency()
    #'simulated_crpsets.csv'
    # Evaluate the models
    challenges, responses = extract_sets(transform=True)
    evaluate_ml_models(challenges, responses)
    find_eighty_percent(challenges, responses)
    detailed_model_scores(challenges, responses)


if __name__ == "__main__":
    setup_folders()
    main()