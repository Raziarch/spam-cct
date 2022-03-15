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
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Local Packages
from evaluate import evaluate_predictions
from lib.utils import support_functions as sf
from load import prepare_ml_data
from lib.utils import modeling_tools as mt

#########################################################################################################
# Setup logging
log = logging.getLogger('info')
#########################################################################################################


# Test Model
def test(args, paths, params, df, unenc_df, models_d, params_d, y_rule):
    log.info('Function Start')
    results_d = {}
    miscopts = params['model']['miscopts']
    config = params['config']
    testopts = params['test']
    transopts = params['load']['transform']
    evalopts = params['evaluate']['test']
    DEBUG = False

    # Convert dataframe to numpy arrays; Make final data adjustments
    X_test, y_test = prepare_ml_data(args, paths, params, df, scope='test')

    # Add noise to the data - Only for validation
    if params['evaluate']['noise_ratio']:
        log.info(sf.Color.BOLD + sf.Color.GREEN +
                 'Beware: Introducing Noise! Ratio: {}'.format(params['evaluate']['noise_ratio']) + sf.Color.END)
        log.info(sf.Color.BOLD + sf.Color.GREEN +
                 'Only used for validation. Data pickles need to be purged before real runs' + sf.Color.END)
        y_test = mt.generate_output_noise(y_test, params['evaluate']['noise_ratio'])

    for m_type in models_d:
        y_test_pred = models_d[m_type].predict_proba(X_test)[:, 1]
        if y_rule is not None:
            log.info('Applying rule-based prediction override....')
            mask_model_tmp = np.array((y_test_pred <= testopts['probability_threshold']).astype(int))
            y_rule = y_rule & mask_model_tmp
            mask = -1 * (y_rule - 1)
            y_test_pred = y_rule + mask * y_test_pred
        if DEBUG:
            results = y_test_pred
            results_path = os.path.join(paths['SRCDIR'], 'results.pk')
            results_file = open(results_path, 'wb')
            pickle.dump(results, results_file)
        test_result_df = pd.DataFrame({'y_actual': y_test, 'y_predicted': y_test_pred})

        # Evaluate Predictions
        log.info('Evaluating {}'.format(m_type))
        results = evaluate_predictions(args, paths, params, y_test, test_result_df['y_predicted'],
                                       testopts['probability_threshold'], scope='test')
        if evalopts['log_results']:
            y_pred_bool = 1 * (test_result_df['y_predicted'] > testopts['probability_threshold'])
            out_file = os.path.join(paths['TESTDIR'], m_type + '_results.csv')
            pd.DataFrame({'y': y_pred_bool.values, 'y_prob': test_result_df['y_predicted']}).to_csv(out_file, index=False)

        if config['track_results']:
            remove_features = ['y', transopts['id_column'], 'ts']
            features = [e for e in df.columns.tolist() if e not in remove_features]
            sf.db_append_results(args, paths, params, model_class=m_type, feature_importance=None, results=results, scope='test')

        results_d[m_type] = results.to_dict()
        log.info('run_metrics: test_performance - {}'.format(results_d[m_type]))

        if testopts['debug'] and params['production']['freeze']:
            remove_features = ['y', transopts['id_column']]
            features = [e for e in df.columns.tolist() if e not in remove_features]
            # Dump Predictions Per Model
            result_df = pd.concat([pd.DataFrame(X_test, columns=features),
                                   pd.DataFrame(y_test, columns=['y']),
                                   test_result_df], axis=1)
            out_file = os.path.join(paths['TESTDIR'], m_type + '_test_output.csv')
            result_df.to_csv(out_file)
    log.info('Function End')
    return results_d


def test_model(args, paths, params, test_df, test_unenc_df, models_d, params_d, y_rule):
    log.info('Function Start')
    # Test Model
    os.chdir(paths['TESTDIR'])
    results_d = test(args, paths, params, test_df, test_unenc_df, models_d, params_d, y_rule)
    log.info('Model Prediction Summary (Test): \n{}\n'.format(sf.tabulate_dict(results_d,
                                                                               multilevel=True,
                                                                               transpose=False)))
    log.info('Function End')
    return
