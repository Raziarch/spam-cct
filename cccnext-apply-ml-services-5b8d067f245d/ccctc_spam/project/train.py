################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################


from __future__ import division

# General Packages
import os
import sys
import logging.config
import pickle
import pandas as pd
import numpy as np

# ML Packages
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report

# Local Packages
from lib.utils import support_functions as sf
from project.evaluate import report_feature_importance, build_learning_curve
from lib.utils import modeling_tools as mt
from evaluate import evaluate_predictions
from load import prepare_ml_data

#########################################################################################################
# Setup logging
log = logging.getLogger('info')
#########################################################################################################


# Code example from scikit-learn.org documentation
def param_search(args, paths, params, m_type, df, split):
    log.info('Function Start')
    model = getattr(sys.modules[__name__], m_type)

    miscopts = params['model']['miscopts']
    trainopts = params['train']
    modelopts = params['model']['modelopts'][m_type]

    log.info(sf.Color.BOLD + sf.Color.BLUE + "\nStart Param Search" + sf.Color.END)
    log.info(m_type)
    log.info(modelopts['search'])

    # Perform Parameter Tuning
    if miscopts['run_method'] == 'grid_search':
        clf = GridSearchCV(model(), modelopts['search'], cv=split, **params['param_search']['grid_search'])
    elif miscopts['run_method'] == 'rand_search':
        clf = RandomizedSearchCV(model(), modelopts['search'], cv=split, **params['param_search']['rand_search'])

    X_cv, y_cv = prepare_ml_data(args, paths, params, df, scope='train')
    clf.fit(X_cv, y_cv)

    log.info("Scores on development set:")

    for params, mean_score, scores in clf.grid_scores_:
        log.info("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

    log.info("Best parameters set found on development set:")
    log.info(clf.best_params_)

    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")

    y_true, y_pred = y_cv, clf.predict(X_cv)

    log.info("Detailed Classification Report:")
    log.info(classification_report(y_true, y_pred))

    log.info(sf.Color.BOLD + sf.Color.BLUE + "\nEnd Param Search" + sf.Color.END)
    log.info('Function End')
    return clf


# Train Model
def train(args, paths, params, df, unenc_df):
    log.info('Function Start')
    config = params['config']

    miscopts = params['model']['miscopts']
    trainopts = params['train']
    transopts = params['load']['transform']
    evalopts = params['evaluate']['train']
    testopts = params['test']

    models_d = {}
    params_d = {}
    results_d = {}

    df.reset_index(drop=True, inplace=True)
    split = mt.split_data(args, paths, params, df, scope='train_cv')

    if miscopts['load_best_params']:
        params_d = pickle.load(open(os.path.join(paths['TRAINDIR'], 'params.pkl'), 'rb'))

    for m_type in miscopts['model_type']:

        modelopts = params['model']['modelopts'][m_type]
        model_class = getattr(sys.modules[__name__], m_type)

        if not miscopts['load_best_params']:
            # Do a Static Run with Set Parameters
            if miscopts['run_method'] == 'static':
                log.info(sf.Color.BOLD + sf.Color.BLUE + 'Running {}.....'.format(m_type) + sf.Color.END)
                params_d[m_type] = modelopts['static']
            # Perform Grid (or) Random Search
            elif (miscopts['run_method'] == 'grid_search' or miscopts['run_method'] == 'rand_search') \
                    and not miscopts['load_best_params']:
                log.info(
                    sf.Color.BOLD + sf.Color.BLUE + 'Running {} on {}.....'.format(miscopts['run_method'], m_type) +
                    sf.Color.END)
                log.debug('Search Parameters: \n{}'.format(sf.tabulate_dict(modelopts['search'])))
                clf = param_search(args, paths, params, m_type, df, split)
                log.info('Param Search Results (Top-10): \n\n{}'.format(
                    sf.tabulate_search_results(clf.cv_results_, count=10)))
                params_d[m_type] = clf.best_params_
                log.info('Best Parameters: \n{}'.format(sf.tabulate_dict(params_d[m_type])))

        # If no parameters are passed
        if params_d[m_type] is None:
            params_d[m_type] = {}

        log.info('run_metrics: model_type - {}'.format(m_type))
        log.info('run_metrics: model_params - {}'.format(params_d[m_type]))

        models_d[m_type] = model_class(**params_d[m_type])

        cv_result_df = pd.DataFrame(columns=['y_actual', 'y_predicted'])
        for i, (train_index, cv_index) in enumerate(split):
            log.info('Processing split {}...'.format(i))
            # Convert dataframe to numpy arrays; Make final data adjustments
            log.info('Preparing dataset for training...')
            if trainopts['split_method'] == 'split_by_time':
                df.index = df['ts']
            X_train, y_train = prepare_ml_data(args, paths, params, df.loc[train_index], scope='train')
            log.info('Preparing dataset for cross-validation...')
            X_cv, y_cv = prepare_ml_data(args, paths, params, df.loc[cv_index], scope='train')
            models_d[m_type].fit(X_train, y_train)
            y_cv_pred = models_d[m_type].predict_proba(X_cv)[:, 1]
            cv_result_df = cv_result_df.append(pd.DataFrame({'y_actual': y_cv, 'y_predicted': y_cv_pred}))
            inc_results = evaluate_predictions(args, paths, params, y_cv, y_cv_pred, testopts['probability_threshold'],
                                               scope='train_cv_split_{}'.format(i))
            log.info('Results for fold {}:\n{}\n'.format(i, inc_results))

        # Plot Learning Curves
        if evalopts['learning_curve'] and not params['production']['freeze']:
            X, y = prepare_ml_data(args, paths, params, df, scope='train')
            build_learning_curve(paths, model_class=model_class, x=X, y=y, n_splits=5, **params_d[m_type])

        # Evaluate Predictions
        log.info('Evaluating {}'.format(m_type))
        results = evaluate_predictions(args, paths, params, cv_result_df['y_actual'].tolist(),
                                       cv_result_df['y_predicted'], testopts['probability_threshold'], scope='train')

        if config['track_results']:
            remove_features = ['y', transopts['id_column'], 'ts', 'idx']
            features = [e for e in df.columns.tolist() if e not in remove_features]
            feat_imp_s = mt.get_feature_importance(model=models_d[m_type],
                                                               features=features,
                                                               num_features=10,
                                                               model_class=m_type)
            sf.db_append_results(args, paths, params, model_class=m_type, feature_importance=feat_imp_s, results=results, scope='train')

        results_d[m_type] = results.to_dict()
        log.info('run_metrics: train_performance - {}'.format(results_d[m_type]))

        if trainopts['debug'] and not params['production']['freeze']:
            result_df = pd.concat([df, cv_result_df], axis=1)
            out_file = os.path.join(paths['TRAINDIR'], m_type + '_train_output.csv')
            result_df.to_csv(out_file)

        # Train on full data for production
        if trainopts['retrain']:
            log.info('Retraining model with full data-set in production...')
            X, y = prepare_ml_data(args, paths, params, df, scope='train')
            models_d[m_type].fit(X, y)

    # Report Feature Importance
    remove_features = ['y', transopts['id_column'], 'ts', 'idx']
    features = [e for e in df.columns.tolist() if e not in remove_features]
    report_feature_importance(args, paths, params, models_d, features)
    log.info('Function End')
    return models_d, params_d, results_d


# Execute Training Workflow
def train_model(args, paths, params, train_df, train_unenc_df):

    log.info('Function Start')
    # Isolating Config Parameters
    config = params['config']
    transopts = params['load']['transform']

    os.chdir(paths['TRAINDIR'])
    log.info('Commencing model training.....')
    models_d, params_d, results_d = train(args, paths, params, train_df, train_unenc_df)
    log.info('Model Prediction Summary (Cross Validation): \n{}\n'.format(sf.tabulate_dict(results_d,
                                                                                           multilevel=True,
                                                                                           transpose=False)))
    # Dump Final Model
    if params['production']['freeze']:
        if len(models_d.keys()) == 1:
            model_pickle_file = os.path.join(paths['FINALDIR'], config['model_pickle_file'])
            log.info('Finalizing model: {}....'.format(model_pickle_file))
            pickle.dump(models_d[models_d.keys()[0]], open(model_pickle_file, 'wb'))
        else:
            raise AssertionError('More than one kind of model enabled. Not supported!')
    log.info('Function End')
    return models_d, params_d
