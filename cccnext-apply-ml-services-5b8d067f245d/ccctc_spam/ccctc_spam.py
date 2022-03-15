#!/usr/bin/env python

################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

import os, sys
import logging.config
import pickle
import timeit
import pandas as pd
import numpy as np
from datetime import datetime

# Local Packages
from lib.utils import support_functions as sf
from project.load import load_predict_source, etl_data, transform_source
from project.train import train_model
from project.test import test_model
from project.predict import predict
from project.setup import setup_env, get_args

##################################################################################################################
# Setup Logging
logging.config.fileConfig('./lib/utils/logging.conf', disable_existing_loggers=False)
log = logging.getLogger('info')
##################################################################################################################


# Prediction Fraud Workflow
def predict_fraud(args, paths, params, json_obj=None):
    log.info('Function Start')
    extropts = params['load']['extract']
    transopts = params['load']['transform']

    # Force datatype changes
    if extropts['force_datatypes']:
        # Capture data types for input JSON from YAML
        feats = extropts['base_features']
        dtypes = {k: v for k, v in feats.items() if 'date' not in v}
        # Pandas does not support missing values for integer dtypes
        # As a result, we change int to float
        bool_cols = []
        for key, value in dtypes.items():
            if value == "int":
                if key != transopts['id_column']:
                    dtypes[key] = "float"
            elif value == 'bool' and key != transopts['y_column']:
                bool_cols.append(key)
                del dtypes[key]
        del dtypes[transopts['y_column']]
        # confirmed_fraud field is not expected as an input to the prediction service
        # It should not be included in the data type conversion.
        df = pd.DataFrame(json_obj, dtype='str')
        for bool_col in bool_cols:
            df[bool_col] = df[bool_col].fillna(np.nan)
        df = df.astype(dtypes)
        df = df.replace(to_replace=[r'^None$', r'^ *$'], value=[np.nan, np.nan], regex=True)
    else:
        df = pd.DataFrame(json_obj)
    # Convert JSON to dataframe
    # Load Text Data
    df, unenc_df, y_rule = transform_source(args, paths, params, df, scope='predict')
    # Call Predict
    pred_df = predict(args, paths, params, df, unenc_df, y_rule)
    log.info('Function End')
    return pred_df


# Train Fraud Workflow
def train_fraud(args, paths, params):
    # Extract, Transform Train/Test Data
    stage_start = timeit.default_timer()
    train_df, train_unenc_df, test_df, test_unenc_df, test_y_rule = etl_data(args, paths, params)
    log.info('run_metrics: etl_duration - {}'.format(timeit.default_timer() - stage_start))
    if args.train:
        stage_start = timeit.default_timer()
        params['MODEL_VERSION'] = params['RUN_TSTMP']
        models_d, params_d = train_model(args, paths, params, train_df, train_unenc_df)
        log.info('run_metrics: train_duration - {}'.format(timeit.default_timer() - stage_start))
        if not params['production']['freeze']:
            log.info('Saving models, params.....')
            pickle.dump(models_d, open(os.path.join(paths['TRAINDIR'], 'models.pkl'), 'wb'))
            pickle.dump(params_d, open(os.path.join(paths['TRAINDIR'], 'params.pkl'), 'wb'))
    if args.test:
        if not args.train:
            log.info('Loading saved models, params....')
            models_d = pickle.load(open(os.path.join(paths['TRAINDIR'], 'models.pkl'), 'rb'))
            params_d = pickle.load(open(os.path.join(paths['TRAINDIR'], 'params.pkl'), 'rb'))
        stage_start = timeit.default_timer()
        test_model(args, paths, params, test_df, test_unenc_df, models_d, params_d, test_y_rule)
        log.info('run_metrics: test_duration - {}'.format(timeit.default_timer() - stage_start))


# Main
if __name__ == "__main__":
    start = timeit.default_timer()
    # Load Arguments
    args = get_args()
    # Setup Environment
    log, paths, params = setup_env(args)
    log.info('run_metrics: timestamp - {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
    # Extract, Transform Train/Test Data
    try:
        train_fraud(args, paths, params)
    except Exception as ex:
        log.fatal(ex, exc_info=True)
    stop = timeit.default_timer()
    log.info('run_metrics: total_duration - {}'.format(stop - start))
