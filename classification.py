# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:15:31 2019

@author: 俊男
"""
# In[] Naive Bayesian Classifier default with Gaussian Kernal
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
import pandas as pd

class NaiveBayesClassifier:
    __classifier = None
    __y_columns = None

    def __init__(self, type="gaussian"):
        algorithm_dict = {
                "bernoulli" : BernoulliNB(),
                "multinomial" : MultinomialNB(),
                "gaussian" : GaussianNB()
                          }
        self.__classifier = algorithm_dict[type]

    @property
    def classifier(self):
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier):
        self.__classifier = classifier

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train.values.ravel())
        self.__y_columns = y_train.columns

        return self

    def predict(self, x_test):
        return pd.DataFrame(self.classifier.predict(x_test), index=x_test.index, columns=self.__y_columns)

# In[] Support Vecor Machine (SVM) Classifier default with Gaussian Radial Basis Function (Gaussian RBF)
from sklearn.svm import SVC
import time

class SVM:
    __classifier = None

    __penalty_C = None
    __kernel = None
    __degree = None
    __gamma = None
    __coef0 = None
    __y_columns = None

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0, random_state=int(time.time())):
        self.__penalty_C = C
        self.__kernel = kernel
        self.__degree = degree
        self.__gamma = gamma
        self.__coef0 = coef0

        self.__classifier = SVC(C=self.__penalty_C,
                                kernel=self.__kernel,
                                degree=self.__degree,
                                gamma=self.__gamma,
                                coef0=self.__coef0,
                                random_state=random_state)

    @property
    def classifier(self):
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier):
        self.__classifier = classifier

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train.values.ravel())
        self.__y_columns = y_train.columns

        return self

    def predict(self, x_test):
        return pd.DataFrame(self.classifier.predict(x_test), index=x_test.index, columns=self.__y_columns)

# In[] Decision Tree
from sklearn.tree import DecisionTreeClassifier
import time

class DecisionTree:
    __classifier = None
    __criterion = None
    __y_columns = None

    def __init__(self, criterion="entropy", random_state=int(time.time())):
        self.__criterion = criterion
        self.__classifier = DecisionTreeClassifier(criterion=self.__criterion, random_state=random_state)

    @property
    def classifier(self):
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier):
        self.__classifier = classifier

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)
        self.__y_columns = y_train.columns
        return self

    def predict(self, x_test):
        return pd.DataFrame(self.classifier.predict(x_test), index=x_test.index, columns=self.__y_columns)

# In[] Random Forest
from sklearn.ensemble import RandomForestClassifier
import time

class RandomForest:
    __classifier = None
    __n_estimators = None
    __criterion = None
    __y_columns = None

    def __init__(self, n_estimators=10, criterion="entropy"):
        self.__n_estimators = n_estimators
        self.__criterion = criterion
        self.__classifier = RandomForestClassifier(n_estimators=self.__n_estimators, criterion=self.__criterion, random_state=int(time.time()))

    @property
    def classifier(self):
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier):
        self.__classifier = classifier

    @property
    def n_estimators(self):
        return self.__n_estimators

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train.values.ravel())
        self.__y_columns = y_train.columns
        return self

    def predict(self, x_test):
        return pd.DataFrame(self.classifier.predict(x_test), index=x_test.index, columns=self.__y_columns)
