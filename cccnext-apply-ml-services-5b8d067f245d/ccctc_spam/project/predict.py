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
import pickle
import pandas as pd
import numpy as np

# Local Packages
from load import prepare_ml_data

#########################################################################################################
# Setup logging
logging.config.fileConfig('./lib/utils/logging.conf', disable_existing_loggers=False)
log = logging.getLogger('info')
#########################################################################################################


# Predict
def predict(args, paths, params, df, unenc_df, y_rule):
    log.info('Function Start')
    config = params['config']

    id = df[params['load']['transform']['id_column']]

    # Prepare ML data
    X = prepare_ml_data(args, paths, params, df, scope='predict')

    # Load Model
    model_pickle_file = os.path.join(paths['FINALDIR'], config['model_pickle_file'])
    model = pickle.load(open(model_pickle_file, 'rb'))

    # Getting Predictions, Classification
    pred_na = model.predict_proba(X)[:, 1]
    
    # Apply rule-based classification
    if y_rule is not None:
        log.info('Applying rule based prediction....')
        mask_model_tmp = np.array((pred_na <= params['test']['probability_threshold']).astype(int))
        y_rule = y_rule * mask_model_tmp
        mask = -1 * (y_rule - 1)
        pred_na = y_rule + mask * pred_na

    pred_df = pd.DataFrame({'pred_fraud_prob': pred_na})
    pred_df['pred_fraud'] = 1 * (pred_df['pred_fraud_prob'] > params['test']['probability_threshold'])
    # Return with App ID as unique identifier
    DEBUG=False
    if DEBUG:
        log.info('Writing df_pred.pk')
        dump_path = os.path.join(paths['SRCDIR'], 'df_pred.pk')
        pickle.dump(pred_df, open(dump_path, 'wb'))
    pred_df['app_id'] = id
    log.info('Function End')
    return pred_df

