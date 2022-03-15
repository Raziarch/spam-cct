################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

from __future__ import division

# General Packages
import os
import logging.config
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ML Packages
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import ShuffleSplit

# Local Packages
from lib.utils import support_functions as sf
from lib.utils import modeling_tools
from lib.visualization import visualization


#########################################################################################################
# Setup logging
log = logging.getLogger('info')
#########################################################################################################


# Report feature importance
def report_feature_importance(args, paths, params, clf_d, features):
    feat_imp_d = {}
    evalopts = params['evaluate']
    for m_type in clf_d:
        if hasattr(clf_d[m_type], 'feature_importances_'):
            feat_imp_s = modeling_tools.get_feature_importance(model=clf_d[m_type],
                                                               features=features,
                                                               num_features=20,
                                                               model_class=m_type)
            log.debug('Feature Importances for {}: \n{}'.format(m_type, feat_imp_s))
            if evalopts['train']['plot_feature_importance']:
                visualization.plot_series(feat_imp_s, kind='barh',
                                          title='{}: Feature Importance'.format(m_type),
                                          xlabel='Coefficient',
                                          ylabel='Features',
                                          loc=paths['TRAINDIR'])
            feat_imp_d[m_type] = feat_imp_s
            log.info('run_metrics: feature_importances: {}'.format(list(feat_imp_d[m_type])))

    if feat_imp_d:
        log.info('Feature Importance Summary: \n{}\n'.format(sf.tabulate_dict(feat_imp_d,
                                                                              multilevel=True,
                                                                              transpose=False)))
    return


# Build Learning Curve
def build_learning_curve(paths, model_class, x, y, n_splits=5, **kwargs):
    log.info('Function Start')
    title = 'Learning Curve - ' + model_class.__name__
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
    plt = modeling_tools.plot_learning_curve(model_class(**kwargs), title, x, y, cv=cv, n_jobs=-1)
    plot_file = os.path.join(paths['TRAINDIR'], model_class.__name__ + '_learning_curve.png')
    plt.savefig(plot_file)
    log.info('Function End')


# Evaluate Model Predictions
def evaluate_predictions(args, paths, params, y_actual, y_predicted, threshold=0.5, scope='train'):

    if scope in params['evaluate']:
        evalopts = params['evaluate'][scope]
    else:
        evalopts = params['evaluate']['train']

    if 'test' in scope:
        filedir = paths['TESTDIR']
    else:
        filedir = paths['TRAINDIR']

    y_predicted_class = 1 * (y_predicted > threshold)
    try:
        auc = roc_auc_score(y_actual, y_predicted)
    except ValueError:
        auc = -1
    confusion = confusion_matrix(y_actual, y_predicted_class)
    recall = recall_score(y_actual, y_predicted_class)
    precision = precision_score(y_actual, y_predicted_class)
    accuracy = accuracy_score(y_actual, y_predicted_class)

    if evalopts['plot_probability_distribution']:
        figname = os.path.join(filedir, 'probability_distribution_' + scope + '.png')
        plt.hist(y_predicted.tolist(), bins=20)
        plt.title('Probability Distribution - {}'.format(scope))
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.savefig(figname)
        plt.clf()

    log.info('\n\nConfusion Matrix: \n{}\n'.format(confusion))

    results = pd.Series([auc, recall, precision, accuracy])
    results.index = ["AUC", "Recall", "Precision", "Accuracy"]

    return results
