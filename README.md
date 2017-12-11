Enron Analysis Using Machine Learning
=====================================

<br>

Introduction
------------

The objective of this project is to select a machine learning model which can be used to help identify people who are suspected of committing corporate fraud. In the case of Enron, company stock price fell from $90 to $1 per share due to financial fraud and corruption. In this exercise, we will use a subset of employee data which contains fincancial information and email communications for employees at Enron. Some records will be flagged as POI (person of interest) and the rest will be non-POI. Our goal is to select a machine learning model to train with the Enron sample data to look for patterns and accurately identify persons of interest. Once the model is tuned and the best combination of features are selected, our model will be tested on a separate subset of the Enron data, i.e. the testing dataset, to validate that our model can be used to identify POIs in other suspected instances of corporate fraud.

If a machine learning model can reliably identify corporate fraud it would be a valuable tool that could assist in criminal investigations. A machine learning model can use its computing power to scan millions of records for patterns of corruption much faster than a human could. It could also make objective and new categories and "follow the money" as it is laundered through various entities.

Data
----

The data provided in this exercise includes 146 records of people who worked at Enron. \#\#\#\# Eighteen people are already identified as Persons of Interest (POI) \#\#\#\# meaning they are suspected of committing corporate fraud. The data includes financial fields pertaining to each person's income from salary, stock and other sources of income from the company. It also includes data about emails sent to/from POIs and non-POIs. The assumptions are that POI likely received a great deal more money than non-POIs and the POIs communicated more frequently with each other than with non-POIs.

Outliers
--------

There were three records which appear to be outliers and which were removed from the dataset: Records containing "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" as names did not appear to have any relevance to the data and were removed. A record for 'LOCKHART EUGENE E' contained 'NaN' for every data point and was also removed. The data was converted from a pkl file to csv so that it can be manually reviewed for accuracy and cleaned if needed. A copy of the cleaned data can be viewed in a spreadsheet by opening file: Data.csv.

Features Used and Selection Process
-----------------------------------

There were 36 features in the original data dictionary. Select\_K\_Best was used to determine the most effective features for getting higher scores. The resulting features are: <br> poi<br> exercised\_stock\_options<br> total\_stock\_value<br> bonus<br> salary<br> deferred\_income<br> long\_term\_incentive<br> restricted\_stock<br> total\_payments<br> shared\_receipt\_with\_poi<br> loan\_advances<br> expenses<br> from\_poi\_to\_this\_person<br> from\_this\_person\_to\_poi<br> director\_fees<br> to\_messages<br> deferral\_payments<br> from\_messages<br> restricted\_stock\_deferred<br>

<br> The number of features was reduced to a total of 19 features, including POI. <br> <br> Computers do well with numbers so I also tried a variation in which I added the log of the numeric fields as well as the ratio of emails between POIs to the data. The financial properties of the data include: <br> bonus<br>deferra\_payments<br>deferred\_income<br>director\_fees<br>exercised\_stock\_options<br>expenses<br> loan\_advances<br> long\_term\_incentive<br>restricted\_stock<br>restricted\_stock\_deferred<br>salary<br>total\_payments<br>total\_stock\_value<br> <br>

Features Added
--------------

<br> We are evaluating two things:<br> 1 - The total amount of money received by certain people vs the norm. POI are likely to have received much more money.<br> 2 - The amount of communication between POIs. The proportion of correspondence (number of emails from/to) POIs is likely to be higher than with non-POIs.

I also added a total income to see who may have gotten much more money than the others. In addition, assuming that POIs are likely to have more contact with other POIs, a calculated field was added to get the ratio of communications between POIs. In order to get the ratio of emails on the same scale as the financial data, columns were added to the financial data which calculate the logarithm of the total payments, salary, bonus, total stock value and exercised stock options features.

#### A total of 18 features were selected using KBbestFeature. The features and their scores are as follows: \#\#\#\# <br>

{ exercised\_stock\_options: 24.8150797332<br> total\_stock\_value: 24.1828986786<br> bonus: 20.7922520472<br> salary: 18.2896840434<br> deferred\_income: 11.4584765793<br> long\_term\_incentive: 9.92218601319<br> restricted\_stock: 9.21281062198<br> total\_payments: 8.77277773009<br> shared\_receipt\_with\_poi: 8.58942073168<br> loan\_advances: 7.18405565829<br> expenses: 6.09417331064<br> from\_poi\_to\_this\_person: 5.24344971337<br> from\_this\_person\_to\_poi: 2.38261210823<br> director\_fees: 2.12632780201<br> to\_messages: 1.64634112944<br> deferral\_payments: 0.224611274736<br> from\_messages: 0.169700947622<br> restricted\_stock\_deferred: 0.0654996529099<br> }<br>

'poi' was added to the head of the feature\_list for a total of 19 features. <br><br> The Number of rows containing NaN values for each feature was: <br> bonus: 62<br> salary: 49<br> deferred\_income: 95<br> long\_term\_incentive: 78<br> restricted\_stock: 34<br> total\_payments: 20<br> shared\_receipt\_with\_poi: 57<br> loan\_advances: 140<br> expenses': 49<br> from\_poi\_to\_this\_person: 57<br> from\_this\_person\_to\_poi: 57<br> director\_fees: 127<br> to\_messages: 57<br> deferral\_payments: 105<br> from\_messages: 57<br> restricted\_stock\_deferred: 126<br> poi: 0<br> } <br>

### There are 2 scenarios in the code to be run: first with the original features and second with the newly added calculated features. The new calculated features which were added are: <br>

##### features\_calculated:

##### total\_payments\_log

##### salary\_log

##### bonus\_log

##### total\_stock\_value\_log

##### exercised\_stock\_options\_log

##### poi\_ratio\_messages

<br><br>

Finally, the selected features were scaled with MinMaxScaler. MinMaxScaler converts the range of values to between 0 and 1 (or -1 to 1 if there are negative values) so that the numeric features can be compared on a level playing field. <br><br>

Parameter Tuning
----------------

When evaluating each model, the initial results did not meet the criteria of at least 0.3 recall and precision. It was necessary to tune the features and parameters. <br><br> SelectKBest was used to identify the optimal features in the original data. New calculated features were added for the log of the numeric features and a ratio of emails to and from POI's. <br><br> Using SKLearn's Pipeline and GridSearchCV classes parameter selection was further optimized.
Pipeline combines model transformers and an estimator into a step process. Next, GridSearchCV was given a range of values for several parameters in order to search for the best estimator over the parameter grid with cross-validation. The output was a list of the best parameters to apply to the test data.

Here are the best results after both automated and manual tuning:
<table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
<caption>
Algorithm Test Results
</caption>
<thead>
<tr>
<th style="text-align:left;">
Features
</th>
<th style="text-align:left;">
Classifier
</th>
<th style="text-align:right;">
Accuracy
</th>
<th style="text-align:right;">
Precision
</th>
<th style="text-align:right;">
Recall
</th>
<th style="text-align:left;">
Accuracy.Tester
</th>
<th style="text-align:left;">
Precision.Tester
</th>
<th style="text-align:left;">
Recall.Tester
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Original Kbest
</td>
<td style="text-align:left;">
Decision Tree with Pipeline
</td>
<td style="text-align:right;">
0.81395
</td>
<td style="text-align:right;">
0.3333
</td>
<td style="text-align:right;">
0.6
</td>
<td style="text-align:left;">
0.8124
</td>
<td style="text-align:left;">
0.27563
</td>
<td style="text-align:left;">
0.25
</td>
</tr>
<tr>
<td style="text-align:left;">
Original Kbest
</td>
<td style="text-align:left;">
Linear SVC
</td>
<td style="text-align:right;">
0.88370
</td>
<td style="text-align:right;">
0.5000
</td>
<td style="text-align:right;">
0.4
</td>
<td style="text-align:left;">
div/0
</td>
<td style="text-align:left;">
div/0
</td>
<td style="text-align:left;">
div/0
</td>
</tr>
<tr>
<td style="text-align:left;">
Original Kbest
</td>
<td style="text-align:left;">
Logistic Regression with PCA
</td>
<td style="text-align:right;">
0.90700
</td>
<td style="text-align:right;">
0.2727
</td>
<td style="text-align:right;">
0.6
</td>
<td style="text-align:left;">
0.73827
</td>
<td style="text-align:left;">
0.25608
</td>
<td style="text-align:left;">
0.5055
</td>
</tr>
<tr>
<td style="text-align:left;">
New Kbest
</td>
<td style="text-align:left;">
Decision Tree
</td>
<td style="text-align:right;">
0.88370
</td>
<td style="text-align:right;">
0.5000
</td>
<td style="text-align:right;">
0.4
</td>
<td style="text-align:left;">
0.80547
</td>
<td style="text-align:left;">
0.24155
</td>
<td style="text-align:left;">
0.21450
</td>
</tr>
<tr>
<td style="text-align:left;">
New Kbest
</td>
<td style="text-align:left;">
Linear SVC
</td>
<td style="text-align:right;">
0.83720
</td>
<td style="text-align:right;">
0.3750
</td>
<td style="text-align:right;">
0.6
</td>
<td style="text-align:left;">
div/0
</td>
<td style="text-align:left;">
div/0
</td>
<td style="text-align:left;">
div/0
</td>
</tr>
<tr>
<td style="text-align:left;">
New Kbest
</td>
<td style="text-align:left;">
Logistic Regression
</td>
<td style="text-align:right;">
0.67440
</td>
<td style="text-align:right;">
0.2000
</td>
<td style="text-align:right;">
0.6
</td>
<td style="text-align:left;">
0.75107
</td>
<td style="text-align:left;">
0.31012
</td>
<td style="text-align:left;">
0.708
</td>
</tr>
</tbody>
</table>
<br><br>

Algorithm Selection
-------------------

<br> A variety of algorithms were tried, including: GaussianNB, DecisionTree, SVM, SVC, LinearSVC, AdaBoost, RandomForest, KNeighbors, and Logistic Regression. Each of these algorithms was run and tuned with PCA and GridSearchCV as well as manually to get the best combination of parameters. Results of the tuning were measured by the accuracy, precision and recall scores. LogisticRegression turned out to have the highest accuracy, precision and recall rates when tested and is the algorithm selected. While some algorithms scored very well on accuracy and precision, recall was lower. <br><br>

Algorithm Tuning
----------------

When tuning an algorithm in machine learning, we aim to select the best parameters in order to optimize performance. In machine learning there are automated processes which select parameters and find the best combinations or parameters. Parameter tuning is important because in machine learning, ultimately, the goal is to have an algorithm that tunes itself.
<br> Tuning or hyperparameter optimization in machine learning involves choosing a set of optimal hyperparameters for a learning algorithm. The best combination of constraints, weights or learning rates needs to be found in order to generalize different data patterns in the training data so that the models perform with high scores when run in cross-validation against the test data. <br> In this project, GridSearchCV has been used to tune the models and come up with the best combination of parameters. Other optimization methods include Bayesian optimization, Random Search, and Gradient-based optimization. Learning algorithms such as decision trees and random forests require the user to set parameters or constraints in the best way possible to get the best results in the training data. In this project, we define acceptable performance as having accuracy, precision and recall scores above 0.3.

Parameter Tuning
----------------

While many classifiers have default parameter values, the parameters can be tuned to improve system performance as evaluated by accuracy, precision and recall scores. There are several ways in which the optimal combination of parameters can be determined.
<br><br> Grid Search Cross Validation (GridSearchCV) uses 3-fold KFold or StratifiedFold. It constructs a grid of all the combinations of parameters, tries out each comgination, and then returns a recommendation of the best combination. Pipeline is useful for chaining together tools into a workflow. <br><br> Principal Component Analysis (PCA) is a fast and flexible unsupervised method for dimensionality reduction in the data. PCA describes the relationship between the features and labels. It involves zeroing out one or more of the smallest principal components, resulting in a lower-dimensional projection of the data that preserves the maximal data variance. Transforming the data to a single dimension means that only the components of the data with the highest variance are included so that the most important relationships are kept. <br><br>

### The best results are from the Logistic Regression algorithm which was tuned and evaluated as follows:

*from sklearn import linear\_model, decomposition, datasets*<br> *from sklearn.pipeline import Pipeline*<br> *from sklearn.model\_selection import GridSearchCV*<br>

*logistic = linear\_model.LogisticRegression()*<br> <br> *params\_lr2 = {*<br> + "logistic\_\_tol":\[10**-10, 10**-20\],<br> + "logistic\_\_C":\[0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20\],<br> + "logistic\_\_class\_weight":\['auto'\],<br> + "rbm\_\_n\_components":\[2,3,4\]<br> *}*

*logistic.fit(features\_train,labels\_train)*

*pred = logistic.predict(features\_test)*<br> *accuracy = accuracy\_score(labels\_test, pred)*<br> *precision = precision\_score(labels\_test, pred)*<br> *recall = recall\_score(labels\_test, pred)*<br>

<br> clf\_winner = Pipeline(steps=\[("scaler", scaler),<br> ("skb", SelectKBest(k=19)),<br> ("clf\_winner", LogisticRegression(tol=0.1, C = 1\*\*19, class\_weight='balanced'))\])

Validation
----------

To validate the performance of each algorithm, accuracy, recall and precision scores were calculated for each. Each category is required to be above 0.3 in order for the algorithm to be considered acceptable. To ensure we trained our model correctly, the data was split into two categories: training and testing. When training the data, we used a subset of the overall data, selected an algorithm and tuned the features until the results were acceptable. We then test that same model on the subset of the overall data which was reserved for testing. If the results do not pass acceptance criteria on the test dataset then it means we have made a mistake. One common mistake is overfitting the data by applying too many features or tuning parameters so that they work well on the training data only.

Our model was tested with a Stratified Shuffle Split cross validation iterator in tester.py in order to create random training test sets of the data. K-Fold cross validation takes all of the labeled data and divides it into batches. The model is then trained on K-1 batches, validated on the last batch and repeated for all permutations. Stratified K-Fold also looks at the relative distribution of the classes. If one class or label appears more than another, stratified k-fold will represent that imbalance when it creates batches.

Evaluation Metrics
------------------

Accuracy, precision and recall were used as the primary evaluation metrics. Since the overall dataset was very small, it is not likely to be a good metric to use on its own. It could be a starting point when applied to a much larger dataset. <br><br> \#\#\#\#\# Accuracy \#\#\#\#\# Accuracy is the number of correct predictions divided by the total numer of predictions. In this project, accuracy tells us how many POIs were classified correctly. <br> \#\#\#\#\# Precision \#\#\#\#\# In this project, a high precision score means that the algorithm correctly guesses that a person is a POI. This measures how certain we are that the person really is a POI. It correctly labels POIs (true positives) with a minimum of false positives. <br><br> \#\#\#\#\# Recall \#\#\#\#\# Recall in this project measures the sensitivity of the model. It tells us how many POIs which are correctly identified divided by the number of POIs that should have been identified. In other words, it is the ratio of the number of correctly identified POIs divided by all POIs. It is the probability that the algorithm will identify someone as a POI if they in fact are.
<br><br> Both precision and accuracy are important because a well performing algorithm must be able to correctly identify people who are actually POIs as POIs and to not incorrectly label people who are not POIs. In this project a precision and recall score above 0.3 is considered acceptable.

<br>

Performance Tuning
------------------

We have to beware of overfitting in which a model is tuned with parameters which perform well on the training dataset, but when applied to the actual test data, they perform poorly. It means that the parameters have been overly customized for the training data. We have to be careful to find a balance in parameter tuning so that the models will generalize well with all data, not just the subset trained with. Another way to put this is signal vs. noise. In predictive modeling, the signal is the true underlying pattern that we want to learn from the given data. Noise occurs when irrelevant data or randomness is in a dataset. A well tuned machine learning algorithm will separate the signal from the noise. A model that has learned the noise instead of the signal is considered to be overfit. <br> The opposite of overfitting is underfitting. Underfitting occurs when the model is too simple, for example if it does not have enough features or is regularzied too much. In order to minimize over or under fitting, the dataset is divided into separate training and test subsets. If a model does much better on the training set than on the test set, then it means there may be overfitting. <br> Cross validation is a useful way to prevent overfitting. Training data is used to generate multile mini train-test splits. These splits are used to tun the model. In standard k-fold cross validation, the algorithm on k-1 folds while using the remaining fold as the test set. <br> Training with more data can sometimes help algorithms to better detect the signal from the noise if the sample data size is larger and the data is clean. <br> Removing irrelevant features can also help to improve performance.

References
----------

<a href=https://www.udacity.com/course/intro-to-machine-learning--ud120>Udacity - Intro to Machine Learning</a>

<a href=http://scikit-learn.org/stable/>SciKit Learn - Machine Learning in Python</a>

<a href=https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html> In Depth: Principal Component Analysis</a>

<a href=https://www.civisanalytics.com/blog/workflows-in-python-using-pipeline-and-gridsearchcv-for-more-compact-and-comprehensive-code/>Workflows in Python</a> <br> <br> <br>
