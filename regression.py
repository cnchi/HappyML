# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:16:22 2019

@author: 俊男
"""

# In[] Define the class for Simple Regressor
from sklearn.linear_model import LinearRegression
import pandas as pd

class SimpleRegressor:
    __regressor = None
    __y_columns = None

    def __init__(self):
        self.__regressor = LinearRegression()

    @property
    def regressor(self):
        return self.__regressor

    def fit(self, x_train, y_train):
        self.__regressor.fit(x_train, y_train)
        self.__y_columns = y_train.columns
        return self

    def predict(self, x_test):
        return pd.DataFrame(self.__regressor.predict(x_test), index=x_test.index, columns=self.__y_columns)

    def r_score(self, x_test, y_test):
        return self.__regressor.score(x_test, y_test)

# In[] Define the class for Multiple Linear Regressor
import statsmodels.api as sm
import copy

class MultipleRegressor:

    def __init__(self) :
        self.__regressor = None
        self.__features = None
        self.__named_features = None

    @property
    def regressor(self):
        return self.__regressor

    @property
    def named_features(self):
        return self.__named_features

    def add_constant(self, exog):
        # This function only support DataFrame
        if isinstance(exog, pd.DataFrame):
            # Check if the column 'const' has been added
            if not ('const' in exog):
                exog = sm.add_constant(exog)
            return exog
        else:
            print("Error: HappyML only supports pandas.DataFrame")
            raise TypeError()

    def fit(self, x_train, y_train):
        # Make sure there is a const column before fitting
        x_train = self.add_constant(x_train)

        # If there is a dimension reduction result, use it
        if self.__features is not None:
            x_train = x_train.iloc[:, self.__features]

        self.__regressor = sm.OLS(exog=x_train, endog=y_train).fit()
        return self

    def predict(self, x_test):
        # Make sure there is a const column before predicting
        x_test = self.add_constant(x_test)

        # If there is a dimension reduction result, use it
        if self.__features is not None:
            x_test = x_test.iloc[:, self.__features]

        return self.__regressor.predict(exog=x_test)

    def backward_elimination(self, x_train, y_train, significance=0.05, verbose=False):
        # Make sure there is a const column before reduction
        x_train = self.add_constant(x_train)

        # Initialize variables
        final_features = [i for i in range(x_train.shape[1])]
        p_values = [1.0 for i in range(x_train.shape[1])]
        this_features = copy.copy(final_features)
        prev_adj_rsquared = float("-inf")
        this_adj_rsquared = 0

        while(True):
            # Show final features first (if verbose)
            if verbose:
                feature_names = [x_train.columns[pos] for pos in final_features]
                print("CUR: {} Adj-RSquared={:.4f}".format(dict(zip(feature_names, ["{:.4f}".format(i) for i in p_values])), prev_adj_rsquared))

            # Load the current chosen columns
            x_opt = x_train.iloc[:, this_features]

            # Fit the model with chosen columns
            self.fit(x_train=x_opt, y_train=y_train)
            this_adj_rsquared = self.__regressor.rsquared_adj
            p_values = self.__regressor.pvalues.tolist()

            # Show trial features (if verbose)
            if verbose:
                feature_names = [x_train.columns[pos] for pos in this_features]
                print("TRY: {} Adj-RSquared={:.4f}".format(dict(zip(feature_names, ["{:.4f}".format(i) for i in p_values])), this_adj_rsquared))

            # If Adjust R-Squared reduced, stop the procedure
            if this_adj_rsquared < prev_adj_rsquared:
                if verbose: print("!!! STOP (Adj RSquared getting lower)\n")
                break
            else:
                final_features = this_features

            # Prepare for next round, get the maximum p-value and compare to significance
            this_features = copy.copy(final_features)
            max_pvalue = max(p_values)
            if max_pvalue > significance:
                max_pvalue_index = p_values.index(max_pvalue)
                del this_features[max_pvalue_index]
                prev_adj_rsquared = this_adj_rsquared
                if verbose: print(">>> GO NEXT (Higher Adj RSquared & has p-value>{})\n".format(significance))
            else:
                if verbose: print("!!! STOP (No more p-value>{})\n".format(significance))
                break

        if verbose:
            feature_names = [x_train.columns[pos] for pos in final_features]
            print("*** FINAL FEATURES: {}".format(feature_names))

        self.__features = final_features
        self.__named_features = [x_train.columns[pos] for pos in final_features if x_train.columns[pos] != "const"]
        return final_features

    def r_score(self):
        return self.__regressor.rsquared_adj

# In[] Define the Class for Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np

class PolynomialRegressor:
    __degree = 1
    __regressor = None
    __poly_regressor = None
    __X_poly = None

    def __init__(self):
        pass

    @property
    def degree(self):
        return self.__degree

    @degree.setter
    def degree(self, degree):
        if degree > 0:
            self.__degree = degree
        else:
            self.__degree = 1

    @property
    def X_poly(self):
        return self.__X_poly

    @property
    def regressor(self):
        return self.__regressor

    @property
    def poly_regressor(self):
        return self.__poly_regressor

    def best_degree(self, x_train, y_train, x_test, y_test, max_degree=10, verbose=False):
        the_best = []
        best_deg, min_rmse = 0, float("inf")

        # Calculate the RMSE of each degree
        for deg in range(1, max_degree+1):
            self.degree = deg
            y_pred = self.fit(x_train, y_train).predict(x_test=x_test)
            this_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            if this_rmse < min_rmse:
                best_deg = deg
                min_rmse = this_rmse

            the_best.append(best_deg)

            if verbose:
                print("Degree {}: RMSE={:.4f} (BEST DEG={}, RMSE={:.4f})".format(deg, this_rmse, best_deg, min_rmse))

        # Get the best degree
        keys_degree, values_freq = np.unique(the_best, return_counts=True)
        degree_freq_dict = dict(zip(keys_degree, values_freq))
        freq_degree_dict = {}
        for k, v in degree_freq_dict.items():
            freq_degree_dict[v] = freq_degree_dict.get(v, [])
            freq_degree_dict[v].append(k)
        max_freq = max(freq_degree_dict)
        best_deg = max(freq_degree_dict[max_freq])

        if verbose:
            print("Frequency vs. Degree dictionary:", freq_degree_dict)
            print("The Best Degree: {}  Frequency: {}".format(best_deg, max_freq))

        self.degree = best_deg
        return self.degree

    def fit(self, x_train, y_train):
        self.__poly_regressor = PolynomialFeatures(self.degree)
        self.__X_poly = pd.DataFrame(self.__poly_regressor.fit_transform(x_train))
        self.__regressor = SimpleRegressor()
        self.__regressor.fit(self.X_poly, y_train)

        return self

    def predict(self, x_test):
        x_test = pd.DataFrame(self.__poly_regressor.fit_transform(x_test), index=x_test.index)
        return self.__regressor.predict(x_test=x_test)

# In[] Define the Class for Logistic Regression
from sklearn.linear_model import LogisticRegression
import time

class LogisticRegressor:
    __regressor = None
    __solver = "lbfgs"
    __y_columns = None

    def __init__(self, solver="lbfgs"):
        if solver not in ("liblinear", "lbfgs", "sag", "saga", "newton-cg"):
            self.__solver = "lbfgs"
        self.__regressor = LogisticRegression(solver=self.solver, random_state=int(time.time()))

    @property
    def regressor(self):
        return self.__regressor

    @property
    def solver(self):
        return self.__solver

    def fit(self, x_train, y_train):
        self.__y_columns = y_train.columns
        if y_train.ndim > 1:
            y_train = y_train.values.ravel()

        self.regressor.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        return pd.DataFrame(self.regressor.predict(x_test), index=x_test.index, columns=self.__y_columns)
