# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:47:15 2017

@author: Yasas
"""
#
# Load libraries
# import pandas
# from pandas.tools.plotting import scatter_matrix
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#
# Input transforms
#
# Spot Check Algorithms
models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#
# evaluate each model in turn
def cross_val(features, labels, seed = 7, scoring = 'accuracy'):
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=20, random_state=seed)
        cv_results = model_selection.cross_val_score(model, features, labels, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)