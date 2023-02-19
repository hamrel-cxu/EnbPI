import pickle
from scipy.stats import skew
import seaborn as sns
import PI_class_EnbPI_journal as EnbPI
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN  # kNN detector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import utils_EnbPI_journal as util
from matplotlib.lines import Line2D  # For legend handles
import calendar
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
import itertools
import importlib
import time
import pandas as pd
import numpy as np
import os
import sys

warnings.filterwarnings("ignore")
importlib.reload(sys.modules["PI_class_EnbPI_journal"])

"""Sec 5.2 Figure 3 on comparing with time-series methods
   Silimar results in the appendix are included as well"""


def big_transform(CA_cities, current_city, one_dim, train_size):
    # Used for California data
    # Next, merge these data (so concatenate X_t and Y_t for one_d or not)
    # Return [X_train, X_test, Y_train, Y_test] from data_x and data_y
    # Data_x is either multivariate (direct concatenation)
    # or univariate (transform each series and THEN concatenate the transformed series)
    big_X_train = []
    big_X_predict = []
    for city in CA_cities:
        data = eval(f"data{city}")  # Pandas DataFrame
        data_x = data.loc[:, data.columns != "DHI"]
        data_y = data["DHI"]
        data_x_numpy = data_x.to_numpy()  # Convert to numpy
        data_y_numpy = data_y.to_numpy()  # Convert to numpy
        X_train = data_x_numpy[:train_size, :]
        X_predict = data_x_numpy[train_size:, :]
        Y_train_del = data_y_numpy[:train_size]
        Y_predict_del = data_y_numpy[train_size:]
        if city == current_city:
            Y_train = Y_train_del
            Y_predict = Y_predict_del
        if one_dim:
            X_train, X_predict, Y_train_del, Y_predict_del = util.one_dimen_transform(
                Y_train_del, Y_predict_del, d=20
            )
            big_X_train.append(X_train)
            big_X_predict.append(X_predict)
            if city == current_city:
                Y_train = Y_train_del
        else:
            big_X_train.append(X_train)
            big_X_predict.append(X_predict)
    X_train = np.hstack(big_X_train)
    X_predict = np.hstack(big_X_predict)
    return [X_train, X_predict, Y_train, Y_predict]


# Read data and initialize parameters
result_type = "Fig3"
response_ls = {
    "Solar_Atl": "DHI",
    "Palo_Alto": "DHI",
    "Wind_Austin": "MWH",
    "green_house": 15,
    "appliances": "Appliances",
    "Beijing_air": "PM2.5",
}
if result_type == "Fig3":
    # Figure 3
    max_data_size = 10000
    dataSolar_Atl = util.read_data(3, "Data/Solar_Atl_data.csv", max_data_size)
    Data_name = ["Solar_Atl"]
    CA_energy_data = False
elif result_type == "AppendixB3":
    # Results in Appendix B.3
    CA_cities = [
        "Fremont",
        "Milpitas",
        "Mountain_View",
        "North_San_Jose",
        "Palo_Alto",
        "Redwood_City",
        "San_Mateo",
        "Santa_Clara",
        "Sunnyvale",
    ]
    for city in CA_cities:
        globals()["data%s" % city] = util.read_CA_data(f"Data/{city}_data.csv")
    Data_name = ["Palo_Alto"]
    CA_energy_data = True
else:
    # Results in Appendix B.4
    datagreen_house = util.read_data(0, "Data/green_house_data.csv", max_data_size)
    dataappliances_data = util.read_data(1, "Data/appliances_data.csv", max_data_size)
    dataBeijing_air = util.read_data(
        2, "Data/Beijing_air_Tiantan_data.csv", max_data_size
    )
    Data_name = ["green_house", "appliances", "Beijing_air"]
min_alpha = 0.0001
max_alpha = 10
ridge_cv = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
random_forest = RandomForestRegressor(
    n_estimators=10, criterion="mse", bootstrap=False, max_depth=2, n_jobs=-1
)
