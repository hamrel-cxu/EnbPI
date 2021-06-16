from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.svm import SVC
from keras.models import Sequential
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from PI_class_ECAD import conformal_AD
from sklearn import neighbors


'''Helper Functions '''


def one_dimen_transform(Y_train, Y_predict, d):
    n = len(Y_train)
    n1 = len(Y_predict)
    X_train = np.zeros((n-d, d))  # from d+1,...,n
    X_predict = np.zeros((n1, d))  # from n-d,...,n+n1-d
    for i in range(n-d):
        X_train[i, :] = Y_train[i:i+d]
    for i in range(n1):
        if i < d:
            X_predict[i, :] = np.r_[Y_train[n-d+i:], Y_predict[:i]]
        else:
            X_predict[i, :] = Y_predict[i-d:i]
    Y_train = Y_train[d:]
    return([X_train, X_predict, Y_train, Y_predict])


def keras_mod():
    model = Sequential(name='NeuralNet')
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    opt = Adam(0.0005)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def keras_rnn():
    model = Sequential(name='RNN')
    # For fast cuDNN implementation, activation = 'relu' does not work
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    opt = Adam(0.0005)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def get_anomalies_classification(data, regr_name, train_size, alpha, stride, dotted,
                                 B_tot=100, rf_num_est=100, phi='mean', neighbor_size=10, return_fitted=False, downsample=True):
    # Note, past_memory only used when one_dim=True
    # Modeling
    # Initialize Parameters
    np.random.seed(98765)
    B = np.random.binomial(B_tot, np.exp(-1))  # number of bootstrap samples
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    data_x_numpy = data_x.to_numpy()  # Convert to numpy
    data_y_numpy = data_y.to_numpy()  # Convert to numpy
    X_train = data_x_numpy[:train_size, :]
    X_predict = data_x_numpy[train_size:, :]
    Y_train = data_y_numpy[:train_size]
    Y_predict = data_y_numpy[train_size:]
    if downsample:
        print('Downsample used')
        train_abnormal = np.where(Y_train == 1)[0]
        train_normal = np.where(Y_train == 0)[0]
        X_train_abnormal = X_train[train_abnormal, :]
        Y_train_abnormal = Y_train[train_abnormal]
        down_sample_idx = np.random.choice(train_normal, 5*len(train_abnormal), replace=False)
        X_train_normal = X_train[down_sample_idx, :]
        Y_train_normal = Y_train[down_sample_idx]
        X_train = np.vstack((X_train_abnormal, X_train_normal))
        Y_train = np.hstack((Y_train_abnormal, Y_train_normal))
    if regr_name == 'Logistic':
        regr = LogisticRegression()
    elif regr_name == 'RF':
        regr = RandomForestClassifier(n_estimators=rf_num_est, criterion='gini',
                                      bootstrap=False, max_depth=3, n_jobs=-1)
    elif regr_name == 'SVC':
        regr = SVC(gamma='auto')
    elif regr_name == 'kNN':
        regr = neighbors.KNeighborsClassifier(n_neighbors=20, weights="distance")
    else:
        raise ValueError('No specification of regressor')
    regr_results = conformal_AD(
        regr, X_train, X_predict, Y_train, Y_predict, neighbor_size, phi)
    # print('fitting model')
    # est_PI, est_nomalies = regr_results.run_experiments(alpha=alpha, B=B, stride=stride, itrial=1)
    # if return_fitted:
    #     print('PIs are returned')
    #     return(est_PI, est_nomalies)
    # else:
    #     print('PIs are *NOT* returned')
    #     return(est_nomalies)
    est_anomalies = regr_results.run_experiments(
        alpha, B, stride, itrial=1, dotted=dotted)
    if return_fitted:
        return(est_anomalies, regr_results)
    return(est_anomalies)


def find_anomalies(p_vals, threshold):
    # Note, this allows threshold to be an array or list with length equal to number
    # of columns of p_vals
    nrow, ncol = p_vals.shape
    anomalies = np.where(p_vals.max(axis=0) <= threshold)[0]
    return(anomalies)

# Check accuracies


def accuracies(estimate, truth):
    precision = sum(np.in1d(estimate, truth))/len(estimate)
    recall = sum(np.in1d(estimate, truth))/len(truth)
    F1 = 2*(precision*recall)/(precision+recall)
    print(f'Precision is {precision}')
    print(f'Recall is {recall}')
    print(f'F1 score is {F1}')
    return(precision, recall, F1)


def plt_prec_recall_F1(df):
    """Plot precision, recall, and F1 score in a row, for all methods
       x-axis is the train_frac
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    methods = np.unique(df['method'])
    train_frac = np.unique(df['train_frac'])
    num_mtds = len(methods)
    axisfont = 22
    tickfont = axisfont - 2
    for j in range(num_mtds):
        mtd = methods[j]
        df_mtd = df.loc[(df.method == mtd) & (df.itrial == 0)]
        precision, recall, F1 = df_mtd['precision'], df_mtd['recall'], df_mtd['F1']
        idx_to_name = {0: precision, 1: recall, 2: F1}
        idx_to_name_1 = {0: 'Precision', 1: 'Recall', 2: r'$F_1$'}
        for i in range(3):
            current_fig = ax[i]
            current_fig.plot(train_frac, idx_to_name[i], label=mtd, linestyle='-',
                             marker='o')
            # Title, labels, ticks, etc.
            current_fig.tick_params(axis='both', which='major', labelsize=tickfont)
            current_fig.set_xticks(train_frac)
            if i == 0:
                current_fig.legend(loc='upper left', bbox_to_anchor=(0.25, 1.45),
                                   ncol=int(num_mtds/2), fontsize=axisfont-2)
            current_fig.set_ylabel(idx_to_name_1[i], fontsize=axisfont)
            current_fig.set_xlabel('Train Fraction', fontsize=axisfont)
    fig.tight_layout(pad=0)
    fig.savefig('Kaggle_plts.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
