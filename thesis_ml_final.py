#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:00:15 2020

@author: Anna
"""

## splitting data & cross validation
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

seed(1)
os.getcwd()

mimicry = pd.read_csv('/Users/Anna/mimicry_degrees.csv', encoding = 'latin1')
mimicry = pd.DataFrame(mimicry)
class_ranking = ['1', '2', '2', '2', '1', '1', '3', '3', 
                 '3', '1', '2', '1', '2', '1', '2', '1', 
                 '2', '1', '2', '1', '2', '3', '3', '3',
                 '1', '2', '2', '3', '1', '1', '2', '3',
                 '3', '3', '3', '3', '3', '3', '3', '3',
                 '3', '3', '2', '1', '2', '2', '1', '3',
                 '3', '3', '3', '1', '2', '1', '1', '3',
                 '1', '2', '2', '2', '3', '3', '3', '1',
                 '1', '1', '1', '2', '2', '3', '3', '3', 
                 '3', '3', '3']
mimicry['class_ranking'] = class_ranking # add classes of the rankings to dataframe
mimicry
X = mimicry.drop(['ï..Pitch', 'Ranking', 'class_ranking'], axis = 1) # delete column with rankings and pitch names
Y = mimicry['class_ranking'] # Y is the column with class rankings

# split data for baseline & testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state = 0) 

# k fold split
kfold = KFold(n_splits = 3, random_state = 0, shuffle = False)

scoring = {'accuracy:', make_scorer(accuracy_score),
           'precision:', make_scorer(precision_score), 
           'recall:', make_scorer(recall_score), 
           'f1_score:', make_scorer(f1_score)}

# for feature scaling the data
scaler = StandardScaler(X, Y) 


## Baseline model 
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)

dummy = DummyClassifier(strategy='uniform', random_state = 1)
dummy.fit(X_train, Y_train)
dummy.score(X_test, Y_test)


## Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty = 'l2', solver = 'lbfgs', random_state = 0, multi_class = 'auto')
model.fit(X_train, Y_train)
model.predict(X_test)
model.score(X_test, Y_test) # ratio correct predictions
confusion_matrix(Y, model.predict(X))

print(classification_report(Y, model.predict(X)))
print(accuracy_score(Y, model.predict(X)))

## K-fold cross validation Logistic Regression
accuracy_kfold = model_selection.cross_val_score(model, X, Y, cv = kfold, scoring = 'accuracy')
Y_pred = cross_val_predict(model, X, Y, cv = kfold)
conf_mat = confusion_matrix(Y, Y_pred)
print(results_kfold)
print(Y_pred)
print(conf_mat)
print("Accuracy:", results_kfold.mean()*100, "%")
print("Precision class 1:", ((1) / (1 + 2 + 10)))
print("Precision class 2:", ((0) / (0 + 0 + 1)))
print("Precision class 3:", ((21) / (21 + 19 + 21)))
print("Recall class 1:", ((1) / (1 + 0 + 21)))
print("Recall class 2:", ((0) / (2 + 0 + 19)))
print("Recall class 3:", ((21) / (10 + 1 + 21)))

## K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier_knn = KNeighborsClassifier(n_neighbors = 25, weights = 'uniform', p = 2)
classifier_knn.fit(X_train, Y_train)

Y_pred = classifier_knn.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(classifier_knn.score(X, Y))
print(accuracy_score(Y_test, Y_pred))

error = []
for i in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'uniform', p = 2)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))

plt.figure(figsize = (11, 8))
plt.plot(range(1, 51), error, color = 'red', linestyle = 'dashed', marker = 'o', 
         markerfacecolor = 'blue', markersize = 10)
plt.title('Error rate')
plt.xlabel('K value')
plt.ylabel('Mean error')

## Kfold cross validation KNN
accuracy_kfold = model_selection.cross_val_score(classifier_knn, X, Y, cv = kfold, scoring = 'accuracy')
Y_pred = cross_val_predict(classifier_knn, X, Y, cv = kfold)
conf_mat = confusion_matrix(Y, Y_pred)
print(results_kfold)
print(Y_pred)
print(conf_mat)
print("Accuracy:", results_kfold.mean()*100, "%")
print("Precision class 1:", ((3) / (3 + 2 + 9)))
print("Precision class 2:", ((1) / (0 + 1 + 3)))
print("Precision class 3:", ((20) / (19 + 18 + 20)))
print("Recall class 1:", ((3) / (3 + 0 + 19)))
print("Recall class 2:", ((1) / (2 + 1 + 18)))
print("Recall class 3:", ((20) / (9 + 3 + 20)))


## Support Vector Machine
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm

classifier_svm.fit(X_train, Y_train)
Y_pred_train = classifier_svm.predict(X_train)
Y_pred = classifier_svm.predict(X_test)

print(accuracy_score(Y_train, Y_pred_train))
print(accuracy_score(Y_test, Y_pred))

## Kfold cross validation SVM
results_kfold = model_selection.cross_val_score(classifier_svm, X, Y, cv = kfold)
Y_pred = cross_val_predict(classifier_svm, X, Y, cv = kfold)
conf_mat = confusion_matrix(Y, Y_pred)
print(results_kfold)
print(Y_pred)
print(conf_mat)
print("Accuracy:", results_kfold.mean()*100, "%")
print("Precision class 1:", ((4) / (4 + 6 + 15)))
#print("Precision class 2:", ((0) / (0 + 0 + 0)))
print("Precision class 3:", ((17) / (18 + 15 + 17)))
print("Recall class 1:", ((4) / (4 + 0 + 18)))
print("Recall class 2:", ((0) / (6 + 0 + 15)))
print("Recall class 3:", ((17) / (15 + 0 + 17)))

pip freeze # version numbers of the packages

## F1 scores

knn1 = ((4) / (4 + 6 + 15))
#knn2 = ((0) / (0 + 0 + 0))
knn3 = ((17) / (18 + 15 + 17))
knn01 = ((4) / (4 + 0 + 18))
knn02 = ((0) / (6 + 0 + 15))
knn03 = ((17) / (15 + 0 + 17))
f1class1knn = 2 * ((knn1 * knn01) / (knn1 + knn01))
f1class2knn = 2 * ((knn2 * knn02) / (knn2 + knn02))
f1class3knn = 2 * ((knn3 * knn03) / (knn3 + knn03))
print(f1class1knn)
print(f1class2knn)
print(f1class2knn)
