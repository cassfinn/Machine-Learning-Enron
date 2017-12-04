#!/usr/bin/python

import sys
import pickle
import csv
import matplotlib.pyplot as plot
import math
from numpy import log
from numpy import sqrt
from math import exp, expm1

sys.path.append("../tools/")

#from email_preprocess import preprocess
from sklearn import cross_validation
from sklearn import svm
from sklearn.model_selection import train_test_split
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, NMF
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import cross_val_score



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
log_features = ["total_payments","salary","bonus","total_stock_value", "exercised_stock_options",]



def make_csv(data_dict):
    """ generates a csv file from the data so we can see the data in a spreadsheet """
    fieldnames = ['name'] + data_dict.itervalues().next().keys()
    with open('data.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data_dict:
            person = data_dict[record]
            person['name'] = record
            assert set(person.keys()) == set(fieldnames)
            writer.writerow(person)


def visualize(data_dict, feature_x, feature_y):
    """ generates a plot of feature y vs feature x, colors poi """

    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])

    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        color = 'red' if poi else 'blue'
        plot.scatter(x, y, color=color)
    plot.xlabel(feature_x)
    plot.ylabel(feature_y)
    plot.show()


def count_invalid_values(data_dict):
    """ counts the number of NaN values for each feature """
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            if person[field] == 'NaN':
                counts[field] += 1
    return counts


def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features



def add_features(data_dict, features_list):
    """
    Given the data dictionary of people with features, adds some features to
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
            #data_dict[name]['poi_ratio_messages_squared'] = poi_ratio ** 2
        except:
            data_dict[name]['poi_ratio_messages'] = 'NaN'

    # print "finished"
    return data_dict


def set_kbest_features_list(data_dict, features_list):
    #get number of NaN values in each feature of data_dict
    num_invalids = count_invalid_values(data_dict)
    print("Number of NaN rows in the data: ")
    pprint(num_invalids)
 

    k_best = get_k_best(data_dict,features_list,len(features_list)-1)
    print("k_best:")
    pprint(k_best)

    import numpy as np
    col_keys = list(k_best.keys())
    arr = list(zip(col_keys))

    poi = ["poi"]
    features_list = poi
    for key in arr:
        features_list += key

    return features_list



stars = 40

print(stars * "*")


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)


### Task: Remove outliers
# these records do not belong
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
# this record has NaN in every field
data_dict.pop('LOCKHART EUGENE E',0)

print(stars * "*")

### Task: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Task: Create new feature(s)
### Store to my_dataset for easy export below.

# combine the poi, financial and email features
features_list = poi + features_financial + features_email

# add calculated columns (log of numeric features and ratio of emails for poi's)
data_dict = add_features(data_dict, features_list)

# recombine the existing and newly calculated fields
features_list = poi + features_calculated + features_financial

### create a csv file to help manually review the updated data_dict as a spreadsheet
make_csv(data_dict)


### get the best features
features_list = set_kbest_features_list(data_dict, features_list)

### store to my_dataset for easy export below
my_dataset = data_dict



print(stars * "*")


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Scale the data so that all data is on a level playing field

scaler = preprocessing.MinMaxScaler()

features = scaler.fit_transform(features)


# fit and transform
pca = PCA()
pca_transform = pca.fit_transform(features)


#features_train, features_test, labels_train, labels_test = preprocess()
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

skb = SelectKBest(k = 'all')

### Task: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Try a variety of classifiers.

print(stars * "*")
print " "


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


print(stars * "*")
print " "

print "GaussianNB with pipeline:"

gnb = GaussianNB()
pipe_GuassianNB =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", gnb)])

pipe_GuassianNB.fit(features_train,labels_train)

pred = pipe_GuassianNB.predict(features_test)
accuracy = pipe_GuassianNB.score(features_test,labels_test)
print "accuracy:",accuracy
print "GaussianNB with pipeline:" 
print("GaussianNB accuracy: ", accuracy)


print(stars * "*")
print(" ")

scaler = preprocessing.MinMaxScaler()
skb = SelectKBest(k = 'all')

print("Decision Tree Classifier")
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

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

print(stars * "*")
print " "

scaler = preprocessing.MinMaxScaler()
skb = SelectKBest(k = 'all')
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

print(stars * "*")
print " "

scaler = preprocessing.MinMaxScaler()
skb = SelectKBest(k = 15)
dt3 = tree.DecisionTreeClassifier(criterion='entropy',splitter='best')
clf_DT3 =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("DecisionTree", dt3)])

clf_DT3.fit(features_train,labels_train)
pred = clf_DT3.predict(features_test)
accuracy =clf_DT3.score(features_test,labels_test)
print "DecisionTree with pipeline Classification report:" 
print "accuracy:",accuracy
accuracy = accuracy_score(labels_test, pred)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)
f1 = f1_score(labels_test, pred)
print("DT-entropy with kbest=10, pipeline accuracy: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1)



print(stars * "*")
print " "

print("svm SVC")
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

print("svm SVC try 1:")
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


print("svm SVC pipeline:")

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

print "results pipeline svm:"
print estimator.best_params_, estimator.best_score_
params = estimator.best_params_

print(stars * "*")
print " "


print("svm SVC try 3:")

clf_svm2 = SVC(C=100, kernel='rbf', decision_function_shape='ovr', degree=3, gamma='auto', coef0=0.0,max_iter=-1, probability=False, random_state=None, shrinking=True)
clf_svm2.fit(features_train,labels_train)
pred = clf_svm2.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print("accuracy SVC: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)

# tune the params
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

print(stars * "*")
print " "


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

print(stars * "*")
print " "


print("AdaBoost")
print(" ")
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


print(stars * "*")
print " "


print("Random Forest")
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(features_train, labels_train)
rf.score(features_train,labels_train)
rf.score(features_test,labels_test)


print(stars * "*")
print " "


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

print(stars * "*")
print " "


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

print " "

KNC2 = KNeighborsClassifier()
pca = PCA()
pipe = Pipeline([('pca', pca), ('knn', KNC2)])
pipe.fit(features_train, labels_train)
pred = pipe.predict(features_test)

print("KNeighborsClassifier - with Pipe:")
print("accuracy KNC2: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)

print(stars * "*")
print " "



print("LinearSVC")
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
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


print(stars * "*")
print " "


print("Logistic Regression")

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


print(stars * "*")
print " "


print("Logistic Regression - Pipeline PCA")

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()

params_lr2 = {
    "logistic__tol":[10**-10, 10**-20],
    "logistic__C":[0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
    "logistic__class_weight":['auto'],
    "rbm__n_components":[2,3,4]
}

logistic.fit(features_train,labels_train)

pred = logistic.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print("accuracy LogisticRegression with PCA: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1)

print(stars * "*")
print " "



### Here is the tuned classifier that generates the highest accuracy, precision and recall scores:
clf_winner = Pipeline(steps=[("scaler", scaler),
                      ("skb", SelectKBest(k=19)),
                      ("clf_winner", LogisticRegression(tol=0.1, C = 1**19, class_weight='balanced'))])

clf_winner.fit(features_train,labels_train)
pred = clf_winner.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)

print("accuracy LogisticRegression with PCA 2: ", accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1)


print(stars * "*")
print " "


dump_classifier_and_data(clf_winner, my_dataset, features_list)



### Testing on tester.py, results:
# Accuracy: 0.75107	Precision: 0.31012	Recall: 0.70800	F1: 0.43131	F2: 0.56343
#	Total predictions: 15000	True positives: 1416	False positives: 3150	False negatives:  584	True negatives: 9850







