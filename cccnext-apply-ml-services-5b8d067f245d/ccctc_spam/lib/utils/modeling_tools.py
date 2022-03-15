################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

#########################################################################################################
#  Description: Collection of functions that help modeling
#
#########################################################################################################
from __future__ import division  # Used in matplotlib

import logging
import logging.config
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from datetime import datetime

# sklearn toolkit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report

from lib.utils import support_functions as sf

#########################################################################################################
# Setup logging
log = logging.getLogger("info")
#########################################################################################################


# Get Top-20 features from model
def get_feature_importance(model, features, num_features=20, model_class=''):
    if model_class == 'LogisticRegression':
        # The current implementation only extracts the coefficients
        # It doesn't provide a feature importance equivalence
        # Placeholder until we identify a better method for LogisticRegression
        # Not a significant concern since we seldom use LogisticRegression
        feature_importance = pd.Series(model.coef_[0], index=features).abs()
    else:
        feature_importance = pd.Series(model.feature_importances_, index=features)
    return feature_importance.sort_values(ascending=False)[:num_features]


# Introduce noise in tagged output data. (Used mainly for validation)
# Currently supports only binary classification
def generate_output_noise(y, ratio):
    log.info('Distribution of True/False Before Noise: {}'.format(np.unique(y, return_counts=True)))
    idx = np.random.randint(0, y.shape[0], int(y.shape[0]*ratio))
    y[idx] = ~y[idx]
    log.info('Distribution of True/False After Noise: {}'.format(np.unique(y, return_counts=True)))
    return y


# Generates noise on a list of features for a percentage of the dataset
# Used to generate noisy data set for simulating poor datasets
# Used to test scenarios where incremental clean data is added to the training set
# Allows us to study model resiliency with clea data on top of noisy data
def generate_input_noise(df, noise_level=1, feats=[], data_perc=0.6, debug=False, pickle_file_root=''):
    if debug:
        outfile = pickle_file_root + '_before.pkl'
        df.to_pickle(outfile)
    for feat in feats:
        # due to the different scales of the features
        # it makes more sense to multiply by noise centered at 1
        # rather than add noise centered at 0:
        log.info('Introducing Input Noise: Noise Level: {}; Percentage Data: {}'.format(noise_level, data_perc))
        np.random.seed(0)
        indexes = np.random.choice(len(df), int(len(df) * data_perc), replace=False)
        np.random.seed(0)
        df.loc[indexes, feat] = df.loc[indexes, feat] * np.random.normal(1, noise_level, len(indexes))
        # df[feat] = df[feat] * np.random.normal(1, noise_level, len(indexes))
    if debug:
        outfile = pickle_file_root + '_after.pkl'
        df.to_pickle(outfile)
    return df


class DistiProb:
    """
    Creates a cumulative data set that can be interpolated to provide frequency value for any given bin number
    """

    def __init__(self, data):
        """
        Instantiates the class

        :param data: data to be cumulated
        :return: object
        """
        if type(data) in [list, tuple]:
            data = np.array(data)
        bins, frequency = self._cumdata(data)
        self.f = interp.interp1d(bins, frequency)
        self.bins = bins
        self.freq = frequency

    def _cumdata(self, data):
        """
        cumulate the data

        :param data: data to be cumulated
        :return: bins in the data, frequency of each bin
        """
        bins = np.unique(data)
        frequency = np.zeros(bins.size)
        totalelem = data.size + 1.0

        for cnt, val in enumerate(bins):
            frequency[cnt] = np.sum(data <= val) / totalelem

        return bins, frequency

    def bin_to_p(self, binval):
        """
        For the given data returns the frequency for any bin(x) value

        :param binval: frequency value(s) for the bins
        :return:
        """
        if type(binval) in [list, tuple]:
            binval = np.array(binval)
        retval = np.zeros(binval.size)

        index = binval <= self.bins[0]
        retval[index] = self.freq[0]

        index1 = binval >= self.bins[-1]
        retval[index1] = self.freq[-1]

        index = np.logical_not(np.logical_or(index, index1))
        retval[index] = self.f(binval[index])
        return retval


# Split data
def split_data(args, paths, params, df, scope='train_cv'):
    trainopts = params['train']
    testopts = params['test']
    if scope == 'train_cv':
        method = trainopts['split_method']
    else:
        method = testopts['split_method']
    if scope == 'train' and trainopts['run_kfold']:
        assert (trainopts['split_method'] == 'random' or
                trainopts['split_method'] is None or
                trainopts['split_method'] == 'split_by_feature'), \
            'KFold cannot be used with methods other than random (default) and split_by_feature'
    if method == 'split_by_feature':
        split = split_by_feature(args, paths, params, df, scope=scope)
    elif method == 'split_by_time':
        split = split_by_time(args, paths, params, df, scope=scope)
    else:
        split = split_random(args, paths, params, df, scope=scope)
    return split


# Split Random
def split_random(args, paths, params, df, scope='train_cv'):
    trainopts = params['train']
    testopts = params['test']
    if scope == 'train_cv':
        test_size = trainopts['split_size']
        test_name = 'cv'
    else:
        test_size = testopts['split_size']
        test_name = 'test'
    log.info('Performing random split on data')
    index = df.index
    if trainopts['run_kfold'] and scope == 'train_cv':
        split = []
        kf = KFold(n_splits=trainopts['n_splits'], shuffle=trainopts['shuffle'])
        for train_index, test_index in kf.split(df):
            split.append((train_index, test_index))
    else:
        if test_size != 0:
            train_df, test_df, train_index, test_index = train_test_split(df, index, test_size=test_size,
                                                                          stratify=df['y'], random_state=42)
        else:
            log.info('{} size split is zero!'.format(test_name))
            train_df, test_df, train_index, test_index = train_test_split(df, index, test_size=test_size,
                                                                          random_state=42)
        log.info('Split Size - train: {}; {}: {}'.format(len(train_index), test_name, len(test_index)))
        split = [(train_index, test_index)]
    return split


# Split by feature value
def split_by_feature(args, paths, params, df, scope='train_cv'):
    trainopts = params['train']
    testopts = params['test']
    if scope == 'train_cv':
        test_size = trainopts['split_size']
        test_name = 'cv'
        feature_col = trainopts['split_by_feature']
    else:
        test_size = testopts['split_size']
        test_name = 'test'
        feature_col = testopts['split_by_feature']
    unique_values = df[feature_col].unique()
    if trainopts['run_kfold'] and scope == 'train_cv':
        split = []
        kf = KFold(n_splits=trainopts['n_splits'], shuffle=trainopts['shuffle'])
        for train_idx, test_idx in kf.split(unique_values):
            train_values = unique_values[train_idx]
            cv_values = unique_values[test_idx]
            log.info('Splitting by feature: {}'.format(feature_col))
            log.info('Feature Values in train: {}'.format(train_values))
            log.info('Feature Values in {}: {}'.format(test_name, cv_values))
            train_index = df.loc[df[feature_col].isin(train_values)].index
            test_index = df.loc[df[feature_col].isin(cv_values)].index
            log.info('Split Size - train: {}; {}: {}'.format(len(train_index), test_name, len(test_index)))
            split.append((train_index, test_index))
    else:
        if test_size != 0:
            train_values, cv_values = train_test_split(unique_values, test_size=test_size, random_state=42)
            log.info('Splitting by feature: {}'.format(feature_col))
            log.info('Feature Values in train: {}'.format(train_values))
            log.info('Feature Values in {}: {}'.format(test_name, cv_values))
            train_index = df.loc[df[feature_col].isin(train_values)].index
            test_index = df.loc[df[feature_col].isin(cv_values)].index
            log.info('Split Size - train: {}; {}: {}'.format(len(train_index), test_name, len(test_index)))
            split = [(train_index, test_index)]
        else:
            log.info('{} size split is zero!'.format(test_name))
            train_df, test_df, train_index, test_index = train_test_split(df, df.index, test_size=test_size,
                                                                          random_state=42)
            log.info('Split Size - train: {}; {}: {}'.format(len(train_index), test_name, len(test_index)))
            split = [(train_index, test_index)]
    return split


# Split by time
def split_by_time(args, paths, params, df, scope='train_cv'):
    trainopts = params['train']
    testopts = params['test']
    if scope == 'train_cv':
        test_size = trainopts['split_size']
        test_name = 'cv'
        time_boundary = trainopts['split_by_time']
    else:
        test_size = testopts['split_size']
        test_name = 'test'
        time_boundary = testopts['split_by_time']
    if test_size != 0:
        df['idx'] = df.index
        df.index = df['ts']
        log.info('Splitting by time. Marker: {}'.format(time_boundary))
        train_index = df[df.index < datetime.fromordinal(time_boundary.toordinal())].index
        test_index = df[df.index >= datetime.fromordinal(time_boundary.toordinal())].index
        log.info('Split Size - Train: {}; {}: {}'.format(len(train_index), test_name, len(test_index)))
        df.index = df['idx']
    else:
        log.info('{} size split is zero! Overriding time-based split'.format(test_name))
        train_df, test_df, train_index, test_index = train_test_split(df, df.index, test_size=test_size,
                                                                      random_state=42)
        log.info('Split Size - Train: {}; {}: {}'.format(len(train_index), test_name, len(test_index)))
    return [(train_index, test_index)]


def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.model_selection module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    log.info('Function Start')
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    log.info('Function End')
    return plt


def plot_validation_curve(estimator, title, x, y, ylim=None, cv=None,
                          n_jobs=1, param_name="n_estimators", param_range=np.arange(10, 5011, 500)):
    """
    Generate a simple plot of the test and training validation curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    param_name : string
        Name of the parameter that will be varied.

    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.model_selection module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    log.info('Function Start')
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("{}".format(param_name))
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(
        estimator, x, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    log.info('Function End')
    return plt

#######################################################################################################


if __name__ == '__main__':
    print('Current Working Directory: %s' % os.getcwd())

    # Setup Logging
    logging_config_file = os.path.join('logging.conf')
    logging.config.fileConfig(logging_config_file, disable_existing_loggers=False)
    log = logging.getLogger('debug')

    a = np.random.randn(10000, 1)
    d = DistiProb(a)
    p = d.bin_to_p(range(-7, 7))
