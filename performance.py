# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:34:21 2019

@author: 俊男
"""

# In[] Warning setting
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# In[] Root of Mean Square Error (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y_real, y_pred):
    return np.sqrt(mean_squared_error(y_real, y_pred))

# In[] R2 (coefficient of determination)
from sklearn.metrics import r2_score

def r2(y_real, y_pred):
    return r2_score(y_real, y_pred)

# In[] Classification Performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

class ClassificationPerformance:
    __y_real = None
    __y_pred = None

    def __init__(self, y_real, y_pred):
        self.__y_real = y_real
        self.__y_pred = y_pred

    def confusion_matrix(self):
        return confusion_matrix(self.__y_real, self.__y_pred)

    def accuracy(self):
        return accuracy_score(self.__y_real, self.__y_pred)

    def recall(self):
        return recall_score(self.__y_real, self.__y_pred, average="macro")

    def precision(self):
        return precision_score(self.__y_real, self.__y_pred, average="macro")

    def f_score(self, beta=1):
        return fbeta_score(self.__y_real, self.__y_pred, beta=beta, average="macro")

# In[] K-Fold Cross Validation of Classification Performance
from sklearn.model_selection import cross_val_score

class KFoldClassificationPerformance:
    __k_fold = None
    __x_ary = None
    __y_ary = None
    __classifier = None
    __verbose = None

    def __init__(self, x_ary, y_ary, classifier, k_fold=10, verbose=False):
        self.__x_ary = x_ary
        self.__y_ary = y_ary
        self.k_fold = k_fold
        self.__classifier = classifier
        self.verbose = verbose

    @property
    def k_fold(self):
        return self.__k_fold

    @k_fold.setter
    def k_fold(self, k_fold):
        if k_fold >=2:
            self.__k_fold = k_fold
        else:
            self.__k_fold = 2

    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, verbose):
        if verbose:
            self.__verbose = 10
        else:
            self.__verbose = 0

    @property
    def classifier(self):
        return self.__classifier

    def accuracy(self):
        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring="accuracy", cv=self.k_fold, verbose=self.verbose)
        return results.mean()

    def recall(self):
        def recall_scorer(estimator, X, y):
            return recall_score(y, estimator.predict(X), average="macro")

        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring=recall_scorer, cv=self.k_fold, verbose=self.verbose)
        return results.mean()

    def precision(self):
        def precision_scorer(estimator, X, y):
            return precision_score(y, estimator.predict(X), average="macro")

        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring=precision_scorer, cv=self.k_fold, verbose=self.verbose)
        return results.mean()

    def f_score(self):
        def f1_scorer(estimator, X, y):
            return fbeta_score(y, estimator.predict(X), beta=1, average="macro")

        results = cross_val_score(estimator=self.classifier, X=self.__x_ary, y=self.__y_ary.values.ravel(), scoring=f1_scorer, cv=self.k_fold, verbose=self.verbose)
        return results.mean()

# In[] GridSearch for Searching the Best Hyper-parameters of a function
from sklearn.model_selection import GridSearchCV

class GridSearch:
    __validator = None
    __estimator = None
    __parameters = None
    __scorer = None
    __k_fold = None
    __best_score = None
    __best_parameters = None
    __best_estimator = None
    __verbose = None

    def __init__(self, estimator, parameters, scorer=None, k_fold=10, verbose=False):
        self.__estimator = estimator
        self.__parameters = parameters
        self.__scorer = scorer
        self.__k_fold = k_fold

        self.verbose = verbose
        self.__validator = GridSearchCV(estimator=self.__estimator, param_grid=self.__parameters, scoring=self.__scorer, cv=self.__k_fold, verbose=self.verbose)

    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, verbose):
        if verbose:
            self.__verbose = 10
        else:
            self.__verbose = 0

    @property
    def validator(self):
        return self.__validator

    @property
    def best_score(self):
        return self.__best_score

    @property
    def best_parameters(self):
        return self.__best_parameters

    @property
    def best_estimator(self):
        return self.__best_estimator

    def fit(self, x_ary, y_ary):
        self.validator.fit(x_ary, y_ary.values.ravel())

        self.__best_parameters = self.validator.best_params_
        self.__best_score = self.validator.best_score_
        self.__best_estimator = self.validator.best_estimator_
