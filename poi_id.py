#!/usr/bin/python

import sys
import pickle
import csv
import matplotlib.pyplot as plot
import math

import helpers_enron

from numpy import log
from numpy import sqrt
from math import exp, expm1

sys.path.append("../tools/")

#from email_preprocess import preprocess
from sklearn import cross_validation
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn import linear_model, decomposition, datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, NMF
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
from helpers_enron import print_separator_line
from pprint import pprint


poi = ["poi"]

### feature of communication between POI's
features_email = [
            "from_messages",
            "from_poi_to_this_person",
            "from_this_person_to_poi",
            "shared_receipt_with_poi",
            "to_messages"
            ]


### Financial features 
features_financial = [
            "bonus",
            "deferral_payments",
            "deferred_income",
            "director_fees",
            "exercised_stock_options",
            "expenses",
            "loan_advances",
            "long_term_incentive",
            "restricted_stock",
            "restricted_stock_deferred",
            "salary",
            "total_payments",
            "total_stock_value"
            ]

features_calculated = [
            "total_payments_log",
            "salary_log",
            "bonus_log",
            "total_stock_value_log",
            "exercised_stock_options_log",
            "poi_ratio_messages",
            ]

### additional features to be added as log of financial values
log_features = ["total_payments","salary","bonus","total_stock_value",]


def get_k_best(data_dict, features_list, k):
    """ 
    Runs scikit-learn's SelectKBest feature selection
    and returns dict where keys=features, values=scores
    """

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k='all')
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features


def add_features(data_dict, features_list):
    """ 
    Given the data dictionary of people with features, adds new features
    for the log of the financial features and ratio of poi:poi emails
    """

    for name in data_dict:
        # add features for the log values of the financial data
        for feat in features_financial:
            try:
                data_dict[name][feat + '_log'] = math.log(data_dict[name][feat] + 1)
            except:
                data_dict[name][feat + '_log'] = 'NaN'

        # Add ratio of POI messages to total.
        try:
            total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                    data_dict[name]["from_this_person_to_poi"] +\
                                    data_dict[name]["shared_receipt_with_poi"]
            poi_ratio = 1.* poi_related_messages / total_messages
            data_dict[name]['poi_ratio_messages'] = poi_ratio
        except:
            data_dict[name]['poi_ratio_messages'] = 'NaN'

    return data_dict


def get_nan_rows(data_dict):
    """ 
    Get the number of NaN values in each feature of data_dict
    """

    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            if person[field] == 'NaN':
                counts[field] += 1

    print("Number of NaN rows in the data: ")
    pprint(counts)
    print_separator_line


def set_kbest_features_list(data_dict, features_list):
    """
    Get the best features
    """
    
    print " --- "

    k_best = get_k_best(data_dict,features_list,len(features_list)-1)
 
    ### Sort the k_best features in descending order
    arr_features = []
    print "sorted k_best:"
    for key, value in sorted(k_best.iteritems(), key=lambda (k,v): (v,k),  reverse=True):
        print "%s: %s" % (key, value)
        arr_features.append(key)

    print "---"
    poi = ["poi"]
    features_list = poi
    features_list += arr_features

    print "---"
    print "features List: "
    print features_list
    print("Number of features in k_best: ",len(features_list))
    print " "

    return features_list


def try_classifier_GaussianNB():
    """ 
    GaussanNB
    """
    
    print "Classifier:  GaussianNB:"
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    clf_GuassianNB = GaussianNB()
    clf_GuassianNB.fit(features_train,labels_train)

    pred = clf_GuassianNB.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)

    print("GaussianNB accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)
    print_separator_line()
    dict_results = { "classifier": "GaussianNB", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_GuassianNB

def try_classifier_GaussianNB_pipeline():
    """
    GaussanNB improved with Pipeline
    """

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print "GaussianNB with pipeline:"
    gnb = GaussianNB()
    skb = SelectKBest(k = 'all')
    pipe_GuassianNB = Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", gnb)])
    pipe_GuassianNB.fit(features_train,labels_train)

    pred = pipe_GuassianNB.predict(features_test)
    accuracy = pipe_GuassianNB.score(features_test,labels_test)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)

    print("GaussianNB with Pipeline, accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)
    print_separator_line()
    dict_results = { "classifier": "GaussianNB with pipeline", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, pred


def try_classifier_Decision_Tree():
    """ 
    Decision Tree Classifier
    """

    print("Decision Tree Classifier, critierion = gini")
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    scaler = preprocessing.MinMaxScaler()
    skb = SelectKBest(k = 'all')
    
    ### Use gini as criterion
    dt = tree.DecisionTreeClassifier(criterion='gini',splitter='best')
    clf_DT1 =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("DecisionTree", dt)])
    clf_DT1.fit(features_train,labels_train)
    pred = clf_DT1.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)

    print("DecisionTree accuracy-pipeline, gini: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)
    print_separator_line()
    dict_results = { "classifier": "Decision Tree, gini", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_DT1


def try_classifier_Decision_Tree2():
    """ 
    Decision Tree Classifier
    """

    print "DecisionTree with criterion = entropy:" 
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    scaler = preprocessing.MinMaxScaler()
    skb = SelectKBest(k = 'all')

    ### Use entropy as criterion
    dt = tree.DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf_DT2 =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("DecisionTree", dt)])
    clf_DT2.fit(features_train,labels_train)
    pred = clf_DT2.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)

    print("DT-entropy with pipeline accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)
    print_separator_line()
    dict_results = { "classifier": "Decision Tree, entropy", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_DT2


def try_classifier_Decision_Tree_Pipeline():
    """ 
    Decision Tree Classifier, optimized with Pipeline
    """
    print "Decision Tree classifier with pipeline:" 
 
    scaler = preprocessing.MinMaxScaler()
    skb = SelectKBest(k = 15)
    dt3 = tree.DecisionTreeClassifier(criterion='entropy',splitter='best')
    clf_DT3 =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("DecisionTree", dt3)])

    clf_DT3.fit(features_train,labels_train)
    pred = clf_DT3.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)

    print("accuracy:",accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)
    print_separator_line()
    print_separator_line()
    dict_results = { "classifier": "Decision Tree, pipeline", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_DT3


def try_svm_classifier():
    """ 
    SVM classifier
    """

    print("svm SVC classifier:")

    clf_svm = svm.SVC(C=10., cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    clf_svm.fit(features_train,labels_train)
    pred = clf_svm.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print("accuracy SVC: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "svm.SVC", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_svm


def try_svc_pipeline_gridsearchcv():
    """
    SVC with pipeline and GridSearchCV
    generate the best parameters
    """

    print "results pipeline svm:"    
    pca = decomposition.PCA()
    svm = SVC()
    pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
    n_components = [10, 14, 18]
    params_grid = {
        'svm__C': [1, 10, 100, 1000],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': [0.001, 0.0001],
        'pca__n_components': n_components,
    }
    estimator = GridSearchCV(pipe, params_grid)
    estimator.fit(features_train,labels_train)

    print estimator.best_params_, estimator.best_score_
    params = estimator.best_params_

    clf_svm2 = SVC(C=100, kernel='rbf', decision_function_shape='ovr', degree=3, gamma='auto', coef0=0.0,max_iter=-1, probability=False, random_state=None, shrinking=True)
    clf_svm2.fit(features_train,labels_train)
    pred = clf_svm2.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print("accuracy SVC: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "svc with pipeline", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_svm2


def try_svc_tuned():
    """ 
    Apply the tuned parameters generated by GridSearchCV
    """

    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel':['rbf'], 'C':[1, 1000]}
    svc = SVC()
    clf_svm3 = GridSearchCV(svc, parameters)
    clf_svm3.fit(features_train,labels_train)
    clf_svm3 = clf_svm3.best_estimator_

    pred = clf_svm3.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print("accuracy SVC: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "svc tuned", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_svm3

    
def try_linear_svc_gridsearchcv():
    """
    Linear SVC model optimized with Pipeline and GridSearchCV
    """

    scaler = preprocessing.MinMaxScaler()
    skb = SelectKBest(k = 'all')

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', LinearSVC())
    ])

    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']
    grid_linearSVC = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)

    grid_linearSVC.fit(features_train,labels_train)
    pred = grid_linearSVC.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print "SVM after applying PCA and GridSearchCV:"
    print("accuracy SVC: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "linear SVC with GridSearchCV", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, grid_linearSVC


def try_linear_svc():
    """
    Linear SVC
    """

    print("LinearSVC")
    from sklearn import svm
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC


    clf_lr = SVC(kernel='rbf', C=1000)
    clf_lr.fit(features_train,labels_train)
    pred = clf_lr.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)

    print("accuracy SVC: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)
    print_separator_line()
    dict_results = { "classifier": "linear svc, rbf", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_lr



def try_ada_boost_decision_tree():
    """ 
    AdaBoost appied to Decision Tree
    """

    print("AdaBoost to Decision Tree")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.grid_search import GridSearchCV

    param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "n_estimators": [10, 30]
                 }

    DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

    ABC = AdaBoostClassifier(base_estimator = DTC)

    grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')

    grid_search_ABC.fit(features_train,labels_train)

    pred = grid_search_ABC.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print("DecisionTree after applying AdaBoost and GridSearchCV:")
    print("accuracy AdaBoost: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "AdaBoost decision tree", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, grid_search_ABC


def try_random_forest():
    """ 
    Random Forest classifier
    """

    print("Random Forest")
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()
    rf.fit(features_train, labels_train)
    score_train = rf.score(features_train,labels_train)
    score_test = rf.score(features_test,labels_test)
    print("score train: ", score_train)
    print("score test: ", score_test)

    clf_rf = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=None, max_features=10)
    clf_rf.fit(features_train,labels_train)
    pred = clf_rf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)

    print("accuracy RandomForest: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1 score: ", f1)
    print_separator_line()
    dict_results = { "classifier": "Random Forest", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_rf



def try_k_neighbors():
    """
    K Nearest Neighbors classifier:
    """

    print "KNeighborsClassifier"
    from sklearn.neighbors import NearestNeighbors
    from sklearn.neighbors import KNeighborsClassifier

    pipe = Pipeline([
        ('classify', KNeighborsClassifier())
    ])

    clf_knn = KNeighborsClassifier(n_neighbors=5)
    params_knn = {}

    KNC = KNeighborsClassifier(n_neighbors=2)
    ABC = AdaBoostClassifier(base_estimator = KNC)

    KNC.fit(features_train,labels_train)
    pred = KNC.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    #f1 = f1_score(labels_test, pred)

    print("KNeighborsClassifier:")
    print("accuracy KNC: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "K Nearest Neighbors", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, KNC


def try_k_neighbors_pipeline():
    """
    K Nearest Neighbors with pipeline:
    """

    print "KNeighborsClassifier with Pipeline"
    from sklearn.neighbors import NearestNeighbors
    from sklearn.neighbors import KNeighborsClassifier

    KNC2 = KNeighborsClassifier()
    pca = PCA()
    pipe = Pipeline([('pca', pca), ('knn', KNC2)])
    pipe.fit(features_train, labels_train)
    pred = pipe.predict(features_test)

    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    print("KNeighborsClassifier - with Pipe:")
    print("accuracy KNC2: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "K nearest neighbors, pipeline", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, pipe


def try_logistic_regression_pipeline():
    """
    Logistic Regression with pipeline:
    """

    print("Logistic Regression with pipeline and PCA:")
    from sklearn import linear_model, decomposition, datasets
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()
    pipe_lr = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    pipe_lr.fit(features_train,labels_train)
    pred_lr = pipe_lr.predict(features_test)

    accuracy = accuracy_score(labels_test, pred_lr)
    precision = precision_score(labels_test,pred_lr)
    recall = recall_score(labels_test,pred_lr)
    f1 = f1_score(labels_test,pred_lr)

    print("accuracy SVC: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "Logistic regression, pca and pipeline", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, pipe_lr


def try_logistic_regression():
    """
    Logistic Regression classifier:
    """

    print("Logistic Regression")
    from sklearn import linear_model, decomposition, datasets
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    logistic = linear_model.LogisticRegression()
    logistic.fit(features_train,labels_train)

    pred = logistic.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print("accuracy LogisticRegression with PCA: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print_separator_line()
    dict_results = { "classifier": "Logistic regression", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, logistic


def try_logistic_regression_tuned():
    """
    Logistic Regression tuned
    """

    clf_winner = Pipeline(steps=[("scaler", scaler),
                          ("skb", SelectKBest(k='all')),
                          ("clf_winner", LogisticRegression(tol=0.1, C = 1**19, class_weight='balanced'))])

    clf_winner.fit(features_train,labels_train)
    pred = clf_winner.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print("accuracy LogisticRegression with PCA 2: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    #return clf_winner
    print_separator_line()
    dict_results = { "classifier": "Logistic regression tuned", "accuracy": accuracy, "precision": precision, "recall": recall }
    return dict_results, clf_winner




### Try all of the classifiers

def run_all_classifiers():
    test_results = []
    classifiers = []

    dict, clf = try_classifier_GaussianNB()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_classifier_GaussianNB_pipeline()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_classifier_Decision_Tree()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_classifier_Decision_Tree2()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_classifier_Decision_Tree_Pipeline()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_classifier_Decision_Tree_Pipeline()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_svm_classifier()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_svc_pipeline_gridsearchcv()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_svc_tuned()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_linear_svc_gridsearchcv()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_linear_svc()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_ada_boost_decision_tree()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_random_forest()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_k_neighbors()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_k_neighbors_pipeline()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_logistic_regression()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = try_logistic_regression_pipeline()
    test_results.append(dict)
    classifiers.append(clf)

    dict, clf = clf_winner = try_logistic_regression_tuned()
    test_results.append(dict)
    classifiers.append(clf)

    return test_results


"""
Processing begins here
"""

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### create a csv file to help manually review the original data_dict as a spreadsheet
from helpers_enron import make_csv
make_csv(data_dict)


### Task: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
### This record has NaN in every field
data_dict.pop('LOCKHART EUGENE E',0)


### Task: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

print_separator_line()

### Combine poi, financial and email features
features_list = poi + features_financial + features_email

### Get the best features
features_list = set_kbest_features_list(data_dict, features_list)

### csv file is written in order to see results as spreadsheet
output_file = "test_results_original.csv"




### ***  UNCOMMENT THESE LINES TO RERUN THE CLASSIFIERS WITH ADDED FEATURES ***
### Additional features: log of financial fields and POI email ratio
###
### Add calculated columns (log of numeric features and ratio of emails for poi's)
data_dict = add_features(data_dict, features_list)
### Recombine the existing and newly calculated fields
features_list = poi + features_calculated + features_financial

### Get the best features
features_list = set_kbest_features_list(data_dict, features_list)

### csv file is written in order to see results as spreadsheet
output_file = "test_results_new_features.csv"



### Task: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Store to my_dataset for easy export below
my_dataset = data_dict

### get number of NaN's in each feature
get_nan_rows(data_dict)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale the data so that all features are evaluated on a similar range
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Fit and transform
pca = PCA()
pca_transform = pca.fit_transform(features)

### Split the data into training and test datasets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### Task: Try a variety of classifiers
test_results = run_all_classifiers()
print "test results:"
for dict in test_results:
    print dict
    print " "


"""
### Write classifier test results to a csv file
### Files: test_results_original.csv and if uncommented above: test_results_new_features.csv
"""
with open(output_file, 'w') as outfile:
   csv_writer = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
   for k in test_results:
       csv_writer.writerow([k])


print_separator_line()


my_dataset = data_dict



### Evaluate Linear SVC against tester.py
dict_results, clf_winner = try_linear_svc()
dump_classifier_and_data(clf_winner, my_dataset, features_list)
#Got a divide by zero when trying out: SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
#Precision or recall may be undefined due to a lack of true positive predicitons.



"""
### Evaluate Decision Tree with Pipeline against tester.py
clf_winner = try_classifier_Decision_Tree_Pipeline_eval()
dump_classifier_and_data(clf_winner, my_dataset, features_list)
#Accuracy: 0.81240   Precision: 0.27563  Recall: 0.25000 F1: 0.26219 F2: 0.25474
#    Total predictions: 15000    True positives:  500    False positives: 1314   False negatives: 1500   True negatives: 11686
"""



### Evaluate Decision Tree 
#dict, clf_winner = try_classifier_Decision_Tree2()
#    Accuracy: 0.80547   Precision: 0.24155  Recall: 0.21450 F1: 0.22722 F2: 0.21941
#    Total predictions: 15000    True positives:  429    False positives: 1347   False negatives: 1571   True negatives: 11653



# dict, clf_winner = try_logistic_regression_tuned()
#    Accuracy: 0.75107   Precision: 0.31012  Recall: 0.70800 F1: 0.43131 F2: 0.56343
#    Total predictions: 15000    True positives: 1416    False positives: 3150   False negatives:  584   True negatives: 9850



### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Here is the tuned classifier that generates the highest accuracy, precision and recall scores:
dict, clf_winner = try_logistic_regression_tuned()



dump_classifier_and_data(clf_winner, my_dataset, features_list)





""" 
Testing on tester.py, results:
    Accuracy: 0.75107   Precision: 0.31012  Recall: 0.70800 F1: 0.43131 F2: 0.56343
    Total predictions: 15000    True positives: 1416    False positives: 3150   False negatives:  584   True negatives: 9850
"""

