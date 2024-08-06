# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:52:25 2019

@author: 俊男
"""

# In[] Import Area
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# In[] sample_model(): Draw the model as line, and sample data as scatter
# USAGE: model_drawer.sample_model(sample_data=(X_test, Y_test), model_data=(X_test, Y_pred))

def sample_model(sample_data=None, sample_color="red", model_data=None, model_color="blue", title="", xlabel="", ylabel="", font=""):
    # for showing Chinese characters
    if font != "":
        plt.rcParams['font.sans-serif']=[font]
        plt.rcParams['axes.unicode_minus'] = False

    # Draw for Sample Data with Scatter Chart
    if sample_data != None:
        plt.scatter(sample_data[0], sample_data[1], color=sample_color)

    # Draw for Model with line chart
    if model_data != None:
        plt.plot(model_data[0], model_data[1], color=model_color)

    # Draw for title, xlabel, ylabel
    if sample_data!=None or model_data!=None:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

# In[] classify_result(): Visualize the result of classification
#   X-axis, Y-asix: Feature 1 & Feature 2 of X (e.g. X-axis = Age, Y-axis = Salary)
#   Background Color: The classification of Y_pred (e.g. RED: Y=0, BLUE: Y=1...)
#   Dot: The samples
#   Dot Color: The classification of Y_real (e.g. RED: Y=0, BLUE: Y=1...)
# USAGE: classifiy_result(x=X_train, y=Y_train, classifier=regressor.regressor, title="訓練集樣本點 vs. 模型", xlabel="年齡", ylabel="薪水")
# NOTE1: Make sure you've done "Feature Scaling" before calling this function.  Otherwise the background dots will be too many to out-of-memory.
# NOTE2: This function can only take 2 features to draw.  Make sure you selected "Best 2" features to train the classifier.

def classify_result(x, y, classifier, fg_color=("orange", "blue"), bg_color=("red", "green"), title="", font=""):
    # Get the xlabel & ylabel first
    xlabel = x.columns[0]
    ylabel = x.columns[1]

    # Convert the color strings into hex codes
    # This is for compatibility with ListedColormap when matplotlib >= 3.8.0
    color_hex_codes = {
        'red': '#FF0000',
        'green': '#008000',
        'blue': '#0000FF',
        'yellow': '#FFFF00',
        'cyan': '#00FFFF',
        'magenta': '#FF00FF',
        'black': '#000000',
        'white': '#FFFFFF',
        'orange': '#FFA500',
        'purple': '#800080'
    }

    fg_color = [color_hex_codes[color] for color in fg_color]
    bg_color = [color_hex_codes[color] for color in bg_color]

    # Prepare each dot of background
    x = x.values
    y = y.values
    x_axis_range = np.arange(x[:, 0].min()-1, x[:, 0].max()+1, 0.01)
    y_axis_range = np.arange(x[:, 1].min()-1, x[:, 1].max()+1, 0.01)
    X_background, Y_background = np.meshgrid(x_axis_range, y_axis_range)

    # Limit the range of drawing
    plt.xlim(X_background.min(), X_background.max())
    plt.ylim(Y_background.min(), Y_background.max())

    # Draw the dots of background (as the predicting result)

    # To predict each dots at X_background x Y_background:
    # 1. X_background.ravel() as Row0, Y_background.ravel() as Row1
    # 2. Transpose Row0, Row1, as Column0, Column1
    # 3. Change it as Dataframe before passing it to .predict()
    # 4. After predict, use .values make DataFrame as NDArray
    # 5. Change 1D back to 2D as dimention X_background.shape
    Target_predict = pd.DataFrame(classifier.predict(pd.DataFrame(np.array([X_background.ravel(), Y_background.ravel()]).T))).values.reshape(X_background.shape)
    plt.contourf(X_background, Y_background, Target_predict, alpha=0.75, cmap=ListedColormap(bg_color))

    # Draw the sample data in dots
    # Iterate all types of Targets (e.g. Y_real = 0, Y_real = 1, ...)
    for y_real_index, y_real in enumerate(np.unique(y)):
        row_selector = y.reshape(x.shape[0]) # y.ndim =2, we need 1D array to select rows of X
        plt.scatter(x[row_selector == y_real, 0], x[row_selector == y_real, 1], c=[ListedColormap(fg_color)(y_real_index)], label=y_real)

    # Set the Title & Label
    # for showing Chinese characters
    if font != "":
        plt.rcParams['font.sans-serif']=[font]
        plt.rcParams['axes.unicode_minus'] = False

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.show()

# In[] tree_drawer(): Visualize the result of decision tree
# USAGE:
#   import robert.model_drawer as md
#   from IPython.display import Image
#   cls_name = [Y_mapping[key] for key in sorted(Y_mapping.keys())]
#   graph = md.tree_drawer(classifier=classifier.classifier, feature_names=X_test.columns, target_names=cls_name)
#   Image(graph.create_png())
# Package Installation:
# (1) Install graphviz first: conda install graphviz + pip install graphviz
# (2) Install pydotplus: conda install pydotplus
# (3) Install GraphViz Executable: Go to https://graphviz.gitlab.io/download/ to download
# (4) Restart Spyder

from sklearn import tree
import os

try:
    import pydotplus
except ImportError:
    pass

def tree_drawer(classifier, feature_names=None, target_names=None, graphviz_bin='C:/Program Files (x86)/Graphviz2.38/bin/'):
    os.environ["PATH"] += os.pathsep + graphviz_bin
    dot_data = tree.export_graphviz(classifier, filled=True, feature_names=feature_names, class_names=target_names, rounded=True, special_characters=True)
    return pydotplus.graph_from_dot_data(dot_data)


# In[] cluster_drawer()
import matplotlib.cm as cm

# Shut off the warning messages from matplotlib
# Reference: https://is.gd/Iq1WGw
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def cluster_drawer(x, y, centroids, title="", font=""):
    # Check for x has only two columns
    if x.shape[1] != 2:
        print("ERROR: x must have only two features to draw!!")
        return None

    # Change y from DataFrame to NDArray
    y_ndarray = y.values.ravel()

    # Get how many classes in y
    y_unique = np.unique(y_ndarray)

    # Iterate all classes in y
    colors = cm.rainbow(np.linspace(0, 1, len(y_unique)))
    for val, col in zip(y_unique, colors):
        plt.scatter(x.iloc[y_ndarray==val, 0], x.iloc[y_ndarray==val, 1], s=50, c=col, label="Cluster {}".format(val))

    # Draw Centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="black", marker="^", label="Centroids")

    # Labels & Legends
    # for showing Chinese characters
    if font != "":
        plt.rcParams['font.sans-serif']=[font]
        plt.rcParams['axes.unicode_minus'] = False

    plt.title(title)
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.legend(loc="best")
    plt.show()


# In[] epochs_metrics_plot(): Draw the line plots of metrics for each epoch during Neural Net training
def epochs_metrics_plot(history_dict, keys=(), title=None, xyLabel=[], ylim=(), size=()):
    lineType = ("-", "--", ".", ":")
    if len(ylim)==2:
        plt.ylim(*ylim)
    if len(size)==2:
        plt.gcf().set_size_inches(*size)
    epochs = range(1, len(history_dict[keys[0]])+1)
    for i in range(len(keys)):
        plt.plot(epochs, history_dict[keys[i]], lineType[i])
    if title:
        plt.title(title)
    if len(xyLabel)==2:
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc="best")
    plt.show()


# In[] show_first_n_images(): Show the first N images of dataset.
def show_first_n_images(x_ary, y_real=[], y_pred=[], first_n=5, font_size=18, color_scheme="gray"):
    # Get Current Figure (GCF) & Set Height 15 inches, Width 4 inches
    plt.gcf().set_size_inches(15, 4)

    # Convert y_pred as NumPy NDArray
    y_pred = np.array(y_pred)

    # Iterate the first N images
    for i in range(first_n):
        # each row has first_n sub-images
        ax = plt.subplot(1, first_n, i+1)

        # "gray": black background, "binary": white background
        ax.imshow(x_ary[i], cmap=color_scheme)

        # set sub-image title
        if y_pred.size == 0:
            img_title = "real = {}".format(y_real[i])
        else:
            img_title = "real = {}\npred = {}".format(y_real[i], y_pred[i])
        ax.set_title(img_title, fontsize=font_size)

        # Make X-axis, Y-axis without ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Show all images
    plt.show()