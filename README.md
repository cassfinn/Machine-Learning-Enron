Enron Analysis Using Machine Learning
=====================================

<br>

Introduction
------------

The objective of this project is to select a machine learning model which can be used to help identify people who are suspected of committing corporate fraud. In the case of Enron, company stock price fell from $90 to $1 per share due to financial fraud and corruption. In this exercise, we will use a subset of employee data which contains fincancial information and email communications for employees at Enron. Some records will be flagged as POI (person of interest) and the rest will be non-POI. Our goal is to select a machine learning model to train with the Enron sample data to look for patterns and accurately identify persons of interest. Once the model is tuned and the best combination of features are selected, our model will be tested on a separate subset of the Enron data, i.e. the testing dataset, to validate that our model can be used to identify POIs in other suspected instances of corporate fraud.

If a machine learning model can reliably identify corporate fraud it would be a valuable tool that could assist in criminal investigations. A machine learning model can use its computing power to scan millions of records for patterns of corruption much faster than a human could. It could also make objective and new categories and "follow the money" as it is laundered through various entities.

Data
----

The data provided in this exercise includes 146 records of people who worked at Enron. Eleven people are already identified as Persons of Interest (POI) meaning they are suspected of committing corporate fraud. The data includes financial fields pertaining to each person's income from salary, stock and other sources of income from the company. It also includes data about emails sent to/from POIs and non-POIs. The assumptions are that POI likely received a great deal more money than non-POIs and the POIs communicated more frequently with each other than with non-POIs.

### Outliers

There were three records which appear to be outliers and which were removed from the dataset: Records containing "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" as names did not appear to have any relevance to the data and were removed. A record for 'LOCKHART EUGENE E' contained 'NaN' for every data point and was also removed. The data was converted from a pkl file to csv so that it can be manually reviewed for accuracy and cleaned if needed. A copy of the cleaned data can be viewed in a spreadsheet by opening file: Data.csv.

### Features Used and Selection Process

There were 36 features in the original data dictionary. Select\_K\_Best was used to determine the most effective features for getting higher scores. The resulting features are: <br> \['poi', 'exercised\_stock\_options', 'total\_stock\_value', 'bonus', 'salary', 'deferred\_income', 'long\_term\_incentive', 'restricted\_stock', 'total\_payments', 'shared\_receipt\_with\_poi', 'loan\_advances', 'expenses', 'from\_poi\_to\_this\_person', 'from\_this\_person\_to\_poi', 'director\_fees', 'to\_messages', 'deferral\_payments', 'from\_messages', 'restricted\_stock\_deferred'\] <br><br> The number of features was reduced to a total of 19 features, including POI. <br> <br> Computers do well with numbers so I ran a variation in which I added the log of the numeric fields as well as the ratio of emails between POIs to the data. The financial properties of the data include: <br> "bonus",<br> "deferral\_payments",<br> "deferred\_income",<br> "director\_fees",<br> "exercised\_stock\_options",<br> "expenses",<br> "loan\_advances",<br> "long\_term\_incentive",<br> "restricted\_stock",<br> "restricted\_stock\_deferred",<br> "salary",<br> "total\_payments",<br> "total\_stock\_value" <br>

### Features Added

We are evaluating two things:<br> 1 - The total amount of money received by certain people vs the norm. POI are likely to have received much more money.<br> 2 - The amount of communication between POIs. The proportion of correspondence (number of emails from/to) POIs is likely to be higher than with non-POIs.

I also added a total income to see who may have gotten much more money than the others. In addition, assuming that POIs are likely to have more contact with other POIs, a calculated field was added to get the ratio of communications between POIs. In order to get the ratio of emails on the same scale as the financial data, columns were added to the financial data which calculate the logarithm of the total payments, salary, bonus, total stock value and exercised stock options features.

A total of 18 features were selected using KBbestFeature:<br> { exercised\_stock\_options<br> total\_stock\_value<br> bonus<br> salary<br> deferred\_income<br> long\_term\_incentive<br> restricted\_stock<br> total\_payments<br> shared\_receipt\_with\_poi<br> loan\_advances<br> expenses<br> from\_poi\_to\_this\_person<br> from\_this\_person\_to\_poi<br> director\_fees<br> to\_messages<br> deferral\_payments<br> from\_messages<br> restricted\_stock\_deferred<br> }<br> <br> 'poi' was added to the head of the feature\_list for a total of 19 features. <br><br>

}<br> <br><br> The Number of NaN rows for each was: <br> {<br> bonus: 62<br> salary: 49<br> deferred\_income: 95<br> long\_term\_incentive: 78<br> restricted\_stock: 34<br> total\_payments: 20<br> shared\_receipt\_with\_poi: 57<br> loan\_advances: 140<br> expenses': 49<br> from\_poi\_to\_this\_person: 57<br> from\_this\_person\_to\_poi: 57<br> director\_fees: 127<br> to\_messages: 57<br> deferral\_payments: 105<br> from\_messages: 57<br> restricted\_stock\_deferred: 126<br> poi: 0<br> } <br> <br>

Finally, the selected features were scaled with MinMaxScaler. <br><br><br>

Parameter Tuning
----------------

When evaluating each model, the initial results did not meet the criteria of at least 0.3 recall and precision. It was necessary to tune the parameters. I used SKLearn's Pipeline and GridSearchCV classes to automate parameter selection.
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
Algorithm Selection
-------------------

Several algorithms were tried, including: GaussianNB, DecisionTree, SVM, SVC, LinearSVC, AdaBoost, RandomForest, KNeighbors, and Logistic Regression. Each of these algorithms was run and tuned with PCA and GridSearchCV as well as manually to get the best combination of parameters. Results of the tuning were measured by the accuracy, precision and recall scores. LogisticRegression turned out to have the highest accuracy, precision and recall rates when tested and is the algorithm selected. While some algorithms scored very well on accuracy and precision, recall was lower.

Validation
----------

To validate the performance of each algorithm, accuracy, recall and precision scores were calculated for each. Each category is required to be above 0.3 in order for the algorithm to be considered acceptable. To ensure we trained our model correctly, the data was split into two categories: training and testing. When training the data, we used a subset of the overall data, selected an algorithm and tuned the features until the results were acceptable. We then test that same model on the subset of the overall data which was reserved for testing. If the results do not pass acceptance criteria on the test dataset then it means we have made a mistake. One common mistake is overfitting the data by applying too many features or tuning parameters so that they work well on the training data only.

Our model was tested with a Stratified Shuffle Split cross validation iterator in tester.py in order to create random training test sets of the data.

#### The best results are from the Logistic Regression algorithm which was tuned and evaluated as follows:

*from sklearn import linear\_model, decomposition, datasets*<br> *from sklearn.pipeline import Pipeline*<br> *from sklearn.model\_selection import GridSearchCV*<br>

*logistic = linear\_model.LogisticRegression()*<br> <br> *params\_lr2 = {*<br> + "logistic\_\_tol":\[10**-10, 10**-20\],<br> + "logistic\_\_C":\[0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20\],<br> + "logistic\_\_class\_weight":\['auto'\],<br> + "rbm\_\_n\_components":\[2,3,4\]<br> *}*

*logistic.fit(features\_train,labels\_train)*

*pred = logistic.predict(features\_test)*<br> *accuracy = accuracy\_score(labels\_test, pred)*<br> *precision = precision\_score(labels\_test, pred)*<br> *recall = recall\_score(labels\_test, pred)*<br>

<br> clf\_winner = Pipeline(steps=\[("scaler", scaler),<br> ("skb", SelectKBest(k=19)),<br> ("clf\_winner", LogisticRegression(tol=0.1, C = 1\*\*19, class\_weight='balanced'))\])

Evaluation Metrics
------------------

Accuracy, precision and recall were used as the primary evaluation metrics. Since the overall dataset was very small, it is not likely to be a good metric to use on its own. It could be a starting point when applied to a much larger dataset. Precision is defined as (\# of true positives)/(\# of true positives + \# of false positives). Recall is defined as: (\# of true positives)/(\# of true positive + \# of false negatives).

These metrics are used to see if we are correctly identifying persons of interest. We want to minimize the number of false positives, i.e. the number of people who are incorrectly flagged as being involved in fraud.

References
----------

<a href=https://www.udacity.com/course/intro-to-machine-learning--ud120>Udacity - Intro to Machine Learning</a>

<a href=http://scikit-learn.org/stable/>SciKit Learn - Machine Learning in Python</a>

<br> <br> <br>
