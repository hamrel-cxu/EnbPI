import importlib
import warnings
from utils_EnbPI import generate_bootstrap_samples, strided_app, weighted_quantile
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
import sys
warnings.filterwarnings("ignore")


class conformal_AD():
    '''
        Use Conformal Prediction Classification to do anomaly detection
        We require each predictor to return estimated probabilities and use
        these probabilities to detect actual anomalies
    '''

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict, neighbor_size, phi='mean'):
        '''
            Fit_func: ridge, lasso, linear model, data
            Decision thres is a new thing (because using 1/0 residual can make certain residuals TOO Large)
        '''
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        self.neighbor_size = neighbor_size
        # it will be updated with a list of bootstrap models, fitted on subsets of training data
        self.Ensemble_fitted_func = []
        # it will store residuals e_1, e_2,... from Ensemble
        # It is a list because it may store results for EACH alpha
        self.Ensemble_online_resid = []
        self.p_val = 0
        self.phi = phi

    def fit_bootstrap_models_online_new(self, alpha, B, dotted=False):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train) and calculate predictions on original data X_train
          Return 1-\alpha quantile of each prdiction on self.X_predict, also
          1. update self.Ensemble_fitted_func with bootstrap estimators and
          2. update self.Ensemble_online_resid with LOO online residuals (from training)
          Update:
           Include tilt option (only difference is using a different test data, so just chaneg name from predict to predict_tilt)
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = generate_bootstrap_samples(n, n, B)
        # hold predictions from each f^b
        # let boot_predictions contain both 1 & 0 probabilities
        boot_predictions = np.zeros((2*B, (n+n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        start = time.time()
        abnormal_idx = np.where(self.Y_train == 1)[0]
        # print(f'Abnormal indices in Training data are {abnormal_idx}')
        '''This Loop Trains B bootstrap models and make predictions '''
        for b in range(B):
            # print(f'b={b}')
            model = self.regressor
            if self.regressor.__class__.__name__ == 'Sequential':
                raise ValueError('NN here is NOT classification algo!')
                # callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
                # if self.regressor.name == 'NeuralNet':
                #     model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b]],
                #               epochs=1000, batch_size=100, callbacks=[callback], verbose=0)
                # else:
                #     # This is RNN, mainly have different shape and decrease epochs
                #     model.fit(self.X_train[boot_samples_idx[b], :], self.Y_train[boot_samples_idx[b]],
                #               epochs=100, batch_size=100, callbacks=[callback], verbose=0)
            else:
                how_many = 2
                if sum(self.Y_train[boot_samples_idx[b]] == 1) < how_many:
                    # Replace the first index with 10 random 1s (i.e. abnormal cases)
                    replace_idx = np.random.choice(len(abnormal_idx), size=how_many)
                    boot_samples_idx[b][:how_many] = abnormal_idx[replace_idx]
                    model = model.fit(self.X_train[boot_samples_idx[b], :],
                                      self.Y_train[boot_samples_idx[b]])
                else:
                    model = model.fit(self.X_train[boot_samples_idx[b], :],
                                      self.Y_train[boot_samples_idx[b]])
            boot_predictions[b] = model.predict_proba(np.r_[self.X_train, self.X_predict])[
                :, 0]  # This is for class 0 (normal)
            # # NOTE: this line below uses P(y_hat=1) as the non-conformity score
            # boot_predictions[b] = 0
            # This is for class 1 (anomaly)
            boot_predictions[b+B] = model.predict_proba(np.r_[self.X_train, self.X_predict])[:, 1]
            self.Ensemble_fitted_func.append(model)
            in_boot_sample[b, boot_samples_idx[b]] = True
        if time.time()-start > 10:
            print(f'Getting fitted models took {time.time()-start} secs')
        start = time.time()
        abnormal_idx = np.where(self.Y_train == 1)[0]
        out_sample_prob_diff_mat = np.zeros((n, n1))
        '''This loop calculates leave-i-out non-conformity scores'''
        if n >= 2000:
            selected_idx = np.array([])
            for j in abnormal_idx:
                neighbors = np.arange(max(j-self.neighbor_size, 0), min(j+self.neighbor_size, n))
                selected_idx = np.append(selected_idx, neighbors).astype(int)
                selected_idx = np.unique(selected_idx)
            for i in selected_idx:
                b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
                if self.phi == 'mean':
                    resid_LOO = np.mean(
                        boot_predictions[b_keep+B, i] - boot_predictions[b_keep, i])
                    out_sample_prob_diff_mat[i] = np.mean(
                        boot_predictions[b_keep + B, n:]-boot_predictions[b_keep, n:], axis=0)
                elif self.phi == 'median':
                    resid_LOO = np.median(
                        boot_predictions[b_keep+B, i] - boot_predictions[b_keep, i])
                    out_sample_prob_diff_mat[i] = np.median(
                        boot_predictions[b_keep + B, n:]-boot_predictions[b_keep, n:], axis=0)
                else:
                    resid_LOO = np.max(
                        boot_predictions[b_keep+B, i] - boot_predictions[b_keep, i])
                    out_sample_prob_diff_mat[i] = np.max(
                        boot_predictions[b_keep + B, n:]-boot_predictions[b_keep, n:], axis=0)
                # self.Ensemble_online_resid = np.append(
                #     self.Ensemble_online_resid, resid_LOO)
            if dotted:
                resid_LOO = np.multiply(resid_LOO, self.Y_train[selected_idx])
                # self.Ensemble_online_resid = np.multiply(
                #     self.Ensemble_online_resid, self.Y_train[selected_idx])
            # NOTE: throw away zero rows of out_sample_predict, since only a few of its rows are filled
            out_sample_prob_diff_mat = out_sample_prob_diff_mat[selected_idx, :]
        else:
            for i in range(n):
                # Same code as above.
                b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
                if self.phi == 'mean':
                    resid_LOO = np.mean(boot_predictions[b_keep+B, i] - boot_predictions[b_keep, i])
                    out_sample_prob_diff_mat[i] = np.mean(
                        boot_predictions[b_keep + B, n:]-boot_predictions[b_keep, n:], axis=0)
                elif self.phi == 'median':
                    resid_LOO = np.median(
                        boot_predictions[b_keep+B, i] - boot_predictions[b_keep, i])
                    out_sample_prob_diff_mat[i] = np.median(
                        boot_predictions[b_keep + B, n:]-boot_predictions[b_keep, n:], axis=0)
                else:
                    resid_LOO = np.max(boot_predictions[b_keep+B, i] - boot_predictions[b_keep, i])
                    out_sample_prob_diff_mat[i] = np.max(
                        boot_predictions[b_keep + B, n:]-boot_predictions[b_keep, n:], axis=0)
                # self.Ensemble_online_resid = np.append(
                #     self.Ensemble_online_resid, resid_LOO)
            if dotted:
                resid_LOO = np.multiply(resid_LOO, self.Y_train)
                # self.Ensemble_online_resid = np.multiply(self.Ensemble_online_resid, self.Y_train)
        n = out_sample_prob_diff_mat.shape[0]
        # NOTE, this will be used for returning p_values
        out_sample_prob_diff_mat.sort(axis=0)  # Sort, time-consuming!!
        A = len(alpha)
        out_sample_prob_diff_ls = []
        for i in range(A):
            ind_q = int((1-alpha[i])*n)
            if alpha[i] == 0:
                ind_q -= 1
            out_sample_prob_diff = out_sample_prob_diff_mat[ind_q]  # length n1
            if dotted:
                out_sample_prob_diff_0_1 = np.multiply(out_sample_prob_diff, self.Y_predict)
                self.Ensemble_online_resid.append(out_sample_prob_diff_0_1)
            else:
                self.Ensemble_online_resid.append(out_sample_prob_diff)
            out_sample_prob_diff_ls.append(out_sample_prob_diff)
        if time.time()-start > 10:
            print(f'Getting residuals took {time.time()-start} secs')
        return(resid_LOO, out_sample_prob_diff_ls, n)

        # NOTE: Changed, because I no longer append stuff but dot product with 0,1 entries
        # Doing so also increases computational speed.
    def compute_PIs_Ensemble_online(self, alpha, B, stride, density_est=False, dotted=False):
        # Now f^b and LOO residuals have been constructed from earlier
        # Finish calibrating residuals and make predictions
        # Return predicted centers of length n1
        # NOTE, n is how many residuals I used for calibration & length of sliding window
        resid_LOO, out_sample_prob_diff_ls, n = self.fit_bootstrap_models_online_new(
            alpha, B, dotted)
        print(f'Sliding window has size {n}')
        A = len(alpha)
        est_anomalies_ls = []
        start = time.time()
        for i in range(A):
            ind_q = int((1-alpha[i])*100)
            # Technically, this is not called width anymore
            strided_resids = strided_app(
                np.hstack((resid_LOO, self.Ensemble_online_resid[i][:-stride])), n, stride)
            # TODO: Commented out section below is for FDR control, not needed for now
            # # New: Compute p-values and store them
            if i == 0:
                # Only record one set of p-vals for a specific alpha
                self.p_val = np.array([np.mean(strided_resids[j, ] >= out_sample_prob_diff_ls[i][j])
                                       for j in range(strided_resids.shape[0])])
            # return 0  # Note, need not the width part below for now.
            width = np.percentile(strided_resids, ind_q, axis=-1)
            width = np.repeat(width, stride)  # This is because |width|=T/stride.
            # Some patch work
            l1 = len(out_sample_prob_diff_ls[i])  # Note, this is T1
            l2 = len(width)
            if l1 > l2:
                diff = l1-l2
                width = np.append(width, width[-1-diff:-1])
            if l1 < l2:
                width = width[:l1]
            est_anomalies_ls.append(np.where(out_sample_prob_diff_ls[i] >= width)[
                                    0])  # NOTE, these are abnormal indices, in test time, NOT starting after Y_train
        print(f'Detection over {A} many alpha took {time.time()-start} secs')
        return(est_anomalies_ls)

    def run_experiments(self, alpha, B, stride, itrial,  dotted=False, density_est=False, methods=['Ensemble']):
        '''
            Note, it is assumed that real data has been loaded, so in actual execution,
            generate data before running this
            Default is for running real-data
        '''
        np.random.seed(98765+itrial)
        for method in methods:
            if method == 'Ensemble':
                est_anomalies_ls = self.compute_PIs_Ensemble_online(
                    alpha, B, stride, density_est, dotted)
            else:
                raise ValueError('Other Conformal AD Methods not yet implemented')
        return(est_anomalies_ls)
