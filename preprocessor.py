# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:04:02 2019

@author: 俊男
"""

# In[] Import Area
import numpy as np
import pandas as pd

# In[] Loading Dataset
# USAGE: dataset = preprocessor.dataset("MyCSVFile.csv")

def dataset(file=""):
    if file != "":
        dataset = pd.read_csv(file)
    else:
        dataset = None

    return dataset

# In[] Decomposite Dataset into Independent Variables & Dependent Variables
# USAGE: X, Y = preprocessor.decomposition(dataset, [0, 1, 2, 3], [4])

def decomposition(dataset, x_columns, y_columns=[]):
    X = dataset.iloc[:, x_columns]
    Y = dataset.iloc[:, y_columns]

    if len(y_columns) > 0:
        return X, Y
    else:
        return X

# In[] Processing Missing Data
# USAGE: X[:, 1:3] = preprocessor.missing_data(X[:, 1:3], strategy="mean")
#       strategy= "mean" | "median", | "most_frequent"

from sklearn.impute import SimpleImputer

def missing_data(dataset, strategy="mean"):
    if strategy not in ("mean", "median", "most_frequent"):
        strategy = "mean"

    if (type(dataset) is pd.DataFrame) and (sum(dataset.isnull().sum()) > 0):
        ary = dataset.values
        missing_cols = [i for i, j in enumerate(dataset.isnull().any()) if j]
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imputer = imputer.fit(ary[:, missing_cols])
        ary[:, missing_cols] = imputer.transform(ary[:, missing_cols])
        return pd.DataFrame(ary, index=dataset.index, columns=dataset.columns)
    else:
    	return dataset


# In[] Categorical Data Pre-processing

# Label Encoder
# USAGE: Y = preprocessor.label_encoder(Y, dtype="float64")
from sklearn.preprocessing import LabelEncoder

def label_encoder(ary, mapping=False):
    encoder = LabelEncoder()
    columns = ary.columns
    index = ary.index
    encoder.fit(ary.values.ravel())
    mapper = {k:v for k, v in enumerate(list(encoder.classes_))}
    encoded_ary = pd.DataFrame(encoder.transform(ary.values.ravel()), index=index, columns=columns)

    if mapping:
        return encoded_ary, mapper
    else:
        return encoded_ary

# One Hot Encoder
# USAGE: X = preprocessor.onehot_encoder(X, [1, 3])
def onehot_encoder(ary, columns=[], remove_trap=False):
    df_results = pd.DataFrame()

    # Iterate each column in DataFrame ary
    for i in range(ary.shape[1]):
        # if this column (i) is dummy column
        if i in columns:
            base_name = ary.columns[i]
            this_column = pd.get_dummies(ary.iloc[:, i])
            this_column = this_column.rename(columns={n:"{}_{}".format(base_name, n) for n in this_column.columns})
            # Remove Dummy Variable Trap if needed
            if remove_trap:
                this_column = this_column.drop(this_column.columns[0], axis=1)
        # else this column is normal column
        else:
            this_column = ary.iloc[:, i]
        # Append this column to the Result DataFrame
        df_results = pd.concat([df_results, this_column], axis=1)

    return df_results

# In[] Spliting Training Set vs. Testing Set
# USAGE: X_train, X_test, Y_train, Y_test = preprocessor.split_train_test(X, Y, train_size=0.8)

from sklearn.model_selection import train_test_split
import time

def split_train_test(x_ary, y_ary, train_size=0.75, random_state=int(time.time())):
    return train_test_split(x_ary, y_ary, test_size=(1-train_size), random_state=random_state)

# In[] Feature Scaling
# USAGE:
#   if transform_arys == None:
#       return the fitted Scaler for future use
#       e.g. scaler = preprocessor.feature_scaling(fit_ary=X_train)
#   if transform_arys != None:
#       return a Tuple for scaled dataset
#       e.g. X_train, X_test = preprocessor.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

from sklearn.preprocessing import StandardScaler

def feature_scaling(fit_ary, transform_arys=None):
    scaler = StandardScaler()
    scaler.fit(fit_ary.astype("float64"))

    if type(transform_arys) is tuple:
        result_list = list()
        for ary in transform_arys:
            ary = pd.DataFrame(ary)
            result_list += [pd.DataFrame(scaler.transform(ary.astype("float64")), index=ary.index, columns=ary.columns)]
        return result_list
    else:
        transform_arys = pd.DataFrame(transform_arys)
        return pd.DataFrame(scaler.transform(transform_arys.astype("float64")), index=transform_arys.index, columns=transform_arys.columns)

# In[] Remove Columns
# USAGE: X = preprocessor.remove_columns(X, [0])

def remove_columns(ary, columns=[]):
    return ary.drop(ary.columns[columns], axis=1)

# In[] Add 1 column for constant
import statsmodels.tools.tools as smtools

def add_constant(ary):
    return smtools.add_constant(ary)

# In[] Join two DataFrame (Left join)

def combine(dataset, y_pred):
    return dataset.join(y_pred)

# In[] Feature Selection: SelectKBest
# SelectKBest requires Y, only good for Supervised Learning

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class KBestSelector:
    __selector = None
    __significance = None
    __best_k = None
    __strategy = None

    def __init__(self, significance=0.05, best_k="auto"):
        self.significance = significance

        if type(best_k) is int:
            self.__strategy = "fixed"
            self.best_k = best_k
        else:
            self.__strategy = "auto"
            self.best_k = 1

        self.__selector = SelectKBest(score_func=chi2, k=self.best_k)

    @property
    def selector(self):
        return self.__selector

    @property
    def significance(self):
        return self.__significance

    @significance.setter
    def significance(self, significance):
        if significance > 0:
            self.__significance = significance
        else:
            self.__significance = 0.05

    @property
    def best_k(self):
        return self.__best_k

    @best_k.setter
    def best_k(self, best_k):
        if best_k >= 1:
            self.__best_k = best_k
        else:
            self.__best_k = 1

    # auto=False has been deprecated in version 2021-05-19
    def fit(self, x_ary, y_ary, verbose=False, sort=False):
        #Get the scores of every feature
        kbest = SelectKBest(score_func=chi2, k="all")
        kbest = kbest.fit(x_ary, y_ary)

        # if auto, choose the best K features
        if self.__strategy == "auto":
            sig_ary = np.full(kbest.pvalues_.shape, self.significance)
            feature_selected = np.less_equal(kbest.pvalues_, sig_ary)
            self.best_k = np.count_nonzero(feature_selected == True)

        # if verbose, show additional information
        if verbose:
            print("\nThe Significant Level: {}".format(self.significance))
            p_values_dict = dict(zip(x_ary.columns, kbest.pvalues_))
            print("\n--- The p-values of Feature Importance ---")

            # if sorted, rearrange p-values in ascending order
            if sort:
                name_pvalue = sorted(p_values_dict.items(), key=lambda kv: kv[1])
            else:
                name_pvalue = [(k, v) for k, v in p_values_dict.items()]

            # Show each feature and its p-value
            for k, v in name_pvalue:
                sig_str = "TRUE  <" if v <= self.significance else "FALSE >"
                sig_str += "{:.2f}".format(self.significance)
                print("{} {:.8e} ({})".format(sig_str, v, k))

            # Show how many features have been selected
            print("\nNumber of Features Selected: {}".format(self.best_k))

        # Start to select features
        self.__selector = SelectKBest(score_func=chi2, k=self.best_k)
        self.__selector = self.__selector.fit(x_ary, y_ary)

        return self


    def transform(self, x_ary):
        # indices=True will return an NDArray of integer for selected columns
        cols_kept = self.selector.get_support(indices=True)
        return x_ary.iloc[:, cols_kept]

# In[] Feature Selection: PCA
# PCA does not require Y, also good for non-supervised learning
# NOTE! Do feature scaling before PCA!!

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class PCASelector:
    __selector = None
    __best_k = None
    __info_covered = None
    __strategy = None

    def __init__(self, best_k="auto"):
        if type(best_k) is int:
            self.__strategy = "fixed"
            self.__best_k = best_k
        elif type(best_k) is float:
            self.__strategy = "percentage"
            self.__info_covered = best_k
            self.__best_k = None
        else:
            self.__strategy = "auto"
            self.__best_k = None

        self.__selector = PCA(n_components=self.__best_k)

    @property
    def selector(self):
        return self.__selector

    @property
    def best_k(self):
        return self.__best_k

    def fit(self, x_ary, verbose=False, plot=False):
        # Get information covered by each component
        pca = PCA(n_components=None)
        pca.fit(x_ary)
        info_covered = pca.explained_variance_ratio_
        if verbose:
            print("Information Covered by Each Component:")
            print(info_covered)
            print()

        # Cumulated Coverage of Information
        cumulated_sum = np.cumsum(info_covered)
        info_covered_dict = dict(zip([i+1 for i in range(info_covered.shape[0])], cumulated_sum))
        if verbose:
            print("Cumulated Coverage of Information:")
            for n, c in info_covered_dict.items():
                print("{:3d}: {}".format(n, c))
            print()

        if self.__strategy == "auto":
            scaler = MinMaxScaler(feature_range=(0, info_covered.shape[0]-1))
            scaled_info_covered = scaler.fit_transform(info_covered.reshape(-1, 1)).ravel()
            for i in range(1, scaled_info_covered.shape[0]):
                if (scaled_info_covered[i-1]-scaled_info_covered[i]) < 1:
                    break
            self.__best_k = i
        elif self.__strategy =="percentage":
            current_best = 1
            cummulated_info = 0.0
            for i in info_covered:
                cummulated_info += i
                if cummulated_info < self.__info_covered:
                    current_best += 1
                else:
                    break
            self.__best_k = current_best

        self.__selector = PCA(n_components=self.best_k)
        self.selector.fit(x_ary)

        if verbose:
            print("Strategy: {}".format(self.__strategy))
            print("Select {} components, covered information {:.2%}".format(self.best_k, info_covered_dict[self.best_k]))
            print()

        if plot:
            np.insert(cumulated_sum, 0, 0.0)
            plt.plot(cumulated_sum, color="blue")
            plt.scatter(x=self.best_k, y=cumulated_sum[self.best_k], color="red")
            plt.title("Components vs. Information")
            plt.xlabel("# of Components")
            plt.ylabel("Covered Information (%)")
            plt.show()

        return self

    def transform(self, x_ary):
        X_columns = ["PCA_{}".format(i) for i in range(1, self.best_k+1)]
        return pd.DataFrame(self.selector.transform(x_ary), index=x_ary.index, columns=X_columns)

