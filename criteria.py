# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 11:43:24 2019

@author: 俊男
"""

# In[] Define the Class for Checking Linear Regression Assumption
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas.plotting import autocorrelation_plot
import pandas as pd
import seaborn as sns
import numpy as np

class AssumptionChecker:
    __x_train = None
    __x_test = None
    __y_train = None
    __y_test = None
    __y_pred = None
    __residuals = None

    __x_lim = None
    __y_lim = None
    __heatmap = False

    def __init__(self, x_train, x_test, y_train, y_test, y_pred):
        self.__x_train = x_train
        self.__x_test = x_test
        self.__y_train = y_train
        self.__y_test = y_test

        self.__y_pred = y_pred
        self.__residuals = (self.__y_test.values.ravel() - self.__y_pred.values.ravel())

    @property
    def x_lim(self):
        return self.__x_lim

    @x_lim.setter
    def x_lim(self, x_lim):
        self.__x_lim = x_lim

    @property
    def y_lim(self):
        return self.__y_lim

    @y_lim.setter
    def y_lim(self, y_lim):
        self.__y_lim = y_lim

    @property
    def heatmap(self):
        return self.__heatmap

    @heatmap.setter
    def heatmap(self, heatmap):
        self.__heatmap = heatmap

    def sample_linearity(self):
        print("*** Check for Linearity of Independent to Dependent Variable ***")

        for i in range(self.__x_train.values.shape[1]):
            plt.scatter(self.__x_train.values[:, i], self.__y_train.values, color="red")
            plt.title("Linearity of Column {}".format(self.__x_train.columns[i]))
            plt.xlabel(self.__x_train.columns[i])
            plt.ylabel("".join(self.__y_train.columns))
            plt.show()

    def residuals_normality(self):
        print("*** Check for Normality of Residuals ***")

        stats.probplot(self.__residuals, plot=plt)
        plt.show()

    def residuals_independence(self):
        print("*** Check for Independence of Residuals ***")

        df_res = pd.DataFrame(self.__residuals)
        autocorrelation_plot(df_res)
        plt.show()

    def residuals_homoscedasticity(self, x_lim=None, y_lim=None):
        print("*** Check for Homoscedasticity of Residuals ***")

        if x_lim != None:
            self.x_lim = x_lim
        if y_lim != None:
            self.y_lim = y_lim

        plt.scatter(self.__y_pred, self.__residuals, color="red", label="Y_pred vs. Residuals")

        dimension = self.__y_pred.shape[0]
        residual_mean = self.__residuals.mean()
        plt.plot(self.__y_pred, np.full(dimension, residual_mean), color="blue", label="Residual Mean")

        plt.title("Y_predict vs. Residuals")
        plt.xlabel("Y_predict")
        plt.ylabel("Residuals")
        plt.legend(loc="best")
        if self.x_lim != None:
            plt.xlim(self.x_lim)
        if self.y_lim != None:
            plt.ylim(self.y_lim)
        plt.show()

    def features_correlation(self, heatmap=None):
        print("*** Check for Correlation of Features ***")
        if heatmap != None:
            self.heatmap = heatmap

        df = pd.DataFrame(self.__x_train)
        corr = df.corr().round(4)
        print("--- Features Correlation Matrix ---")
        print(corr)
        if self.heatmap:
            # annot = annotation = True = put number inside matrix
            sns.heatmap(data=corr, annot=True)
            plt.show()

        corr_ary = corr.to_numpy()
        corr_bool = False
        for i in range(corr_ary.shape[0]):
            for j in range(corr_ary.shape[1]):
                if i != j:
                    if corr_ary[i, j] >= 0.8:
                        corr_bool = True
                        print("Correlation Found at[{}, {}] = {}".format(i, j, corr_ary[i, j]))
        if not corr_bool:
            print("No Correlation (>=0.8) Found!")

    def check_all(self):
        self.sample_linearity()
        self.residuals_normality()
        self.residuals_independence()
        self.residuals_homoscedasticity()
        self.features_correlation()
