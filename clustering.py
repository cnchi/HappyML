# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:18:56 2019

@author: 俊男
"""

# In[] KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class KMeansCluster:
    __cluster = None
    __best_k = None
    __max_k = None
    __strategy = None
    __random_state = None
    __centroids = None

    def __init__(self, best_k="auto", max_k=10, random_state=int(time.time())):
        if type(best_k) is int:
            self.__strategy = "fixed"
            self.best_k = best_k
        else:
            self.__strategy = "auto"
            self.best_k = 8

        self.__max_k = max_k
        self.__random_state = random_state

        self.__cluster = KMeans(n_clusters=self.best_k, max_iter=300, n_init=10, init="k-means++", random_state=self.__random_state)

    @property
    def cluster(self):
        return self.__cluster

    @cluster.setter
    def cluster(self, cluster):
        self.__cluster = cluster

    @property
    def best_k(self):
        return self.__best_k

    @best_k.setter
    def best_k(self, best_k):
        if (type(best_k) is int) and (best_k >= 1):
            self.__best_k = best_k
        else:
            self.__best_k = 1

    @property
    def centroids(self):
        return self.__centroids

    def fit(self, x_ary, verbose=False, plot=False):
        if self.__strategy == "auto":
            wcss = []
            for i in range(1, self.__max_k+1):
                kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init="k-means++", random_state=self.__random_state)
                kmeans.fit(x_ary)
                wcss.append(kmeans.inertia_)

            scaler = MinMaxScaler(feature_range=(0, len(wcss)-1))
            wcss_scaled = scaler.fit_transform(np.array(wcss).reshape(-1, 1)).ravel()
            for i in range(1, wcss_scaled.shape[0]):
                if (wcss_scaled[i-1]-wcss_scaled[i]) < 1:
                    break
            self.best_k = i

            if verbose:
                print("The best clusters = {}".format(self.best_k))

            if plot:
                plt.plot(range(1, len(wcss)+1), wcss, color="blue")
                plt.scatter(x=self.best_k, y=wcss[self.best_k], color="red")
                plt.title("The Best Cluster")
                plt.xlabel("# of Clusters")
                plt.ylabel("WCSS")
                plt.show()

        # Fit the Model
        self.cluster = KMeans(n_clusters=self.best_k, random_state=self.__random_state)
        self.cluster.fit(x_ary)
        self.__centroids = self.cluster.cluster_centers_

        return self

    def predict(self, x_ary, y_column="Result"):
        return pd.DataFrame(self.cluster.predict(x_ary), index=x_ary.index, columns=[y_column])
