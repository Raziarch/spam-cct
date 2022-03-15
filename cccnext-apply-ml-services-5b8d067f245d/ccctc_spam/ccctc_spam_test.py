#!/usr/bin/env python

################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

import os
import errno
import pandas as pd
import numpy as np
import logging.config
import json

from project.setup import load_paths
from ccctc_spam import predict_fraud
from project.setup import get_args, setup_env
from project.evaluate import evaluate_predictions
from lib.utils import support_functions as sf

##################################################################################################################
# Setup Logging
logging.config.fileConfig('./lib/utils/logging.conf', disable_existing_loggers=False)
log = logging.getLogger('info')
##################################################################################################################


# Force a symbolic link
def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )


def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )


def _byteify(data, ignore_dicts=False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.items()
        }
    # if it's anything else, return it in its original form
    return data


# Pre-Split some data out for prediction testing
def generate_prediction_split(paths):
    input_file = os.path.join(paths['SRCDIR'], 'phase1_data.csv')
    train_file = os.path.join(paths['SRCDIR'], 'phase1_data_train.csv')
    train_symlink_file = os.path.join(paths['SRCDIR'], 'ccctc_spam.csv')
    predict_file = os.path.join(paths['SRCDIR'], 'phase1_data_predict.csv')
    log.info('Reading Input CSV file')
    df = pd.read_csv(input_file)
    train_df = df.sample(frac=0.995, random_state=42)
    predict_df = df.drop(train_df.index)
    log.info('Writing Training CSV file')
    train_df.to_csv(train_file, index=False)
    log.info('Writing Prediction CSV file')
    predict_df.to_csv(predict_file, index=False)
    symlink_force(train_file, train_symlink_file)


def create_time_based_quantile_split(df, slice_split):

    """Create a time based split - split on a week and predict the next three weeks"""

    temp = df.groupby(["year", "week", "slice"])["confirmed_fraud"].agg(["count", "mean"])
    temp = temp.reset_index().reset_index()
    temp["split"] = temp["index"]

    df = pd.merge(temp, df, on=["year", "week"])
    df["slice"] = df["index"]

    tr_name = "{slice_split}_weekly_train.csv".format(slice_split=slice_split)
    test_name = "{slice_split}_weekly_test.csv".format(slice_split=slice_split)

    final_train = df[(df["slice"] <= slice_split)]
    final_test = df[(df["slice"] > slice_split) & (df["slice"] <= slice_split + 2)]

    print("Train Shape:{0}".format(final_train.shape))
    print("Test Shape:{0}".format(final_test.shape))

    final_train.to_csv("/home/ubuntu/data/src/phase1.5/2018.03.05/splits/{tr_name}".format(tr_name=tr_name),
                       index=None)
    final_test.to_csv("/home/ubuntu/data/src/phase1.5/2018.03.05/splits/{test_name}".format(test_name=test_name),
                      index=None)


def create_college_time_split(df, perc_split=0.8):
    """
    Chronological split by given category
    perc_split - for each category (college in this case) keep perc_split for train set
    """
    tr_name = "{perc_split}_college_train.csv".format(perc_split=int(100*perc_split))
    test_name = "{perc_split}_college_test.csv".format(perc_split=int(100*perc_split))

    college_ids = df["college_id"].unique()

    train_list = []
    test_list = []
    for college_id in college_ids:
        print(college_id)
        dfc = df[df["college_id"] == college_id]
        print(dfc.shape)
        dfc = dfc.sort_values("tstmp_submit")
        train, test = np.split(dfc, [int(perc_split*len(dfc))])
        train_list.append(train)
        test_list.append(test)

    final_train = pd.concat(train_list)
    final_test = pd.concat(test_list)

    print("Train Shape:{0}".format(final_train.shape))
    print("Test Shape:{0}".format(final_test.shape))

    final_train.to_csv("/home/ubuntu/data/src/phase1.5/2018.03.05/splits/{tr_name}".format(tr_name=tr_name),
                       index=None)
    final_test.to_csv("/home/ubuntu/data/src/phase1.5/2018.03.05/splits/{test_name}".format(test_name=test_name),
                      index=None)


def run_evaluation_splits(paths):

    runs = 0
    do_time_splits = False
    do_noise_splits = True

    if do_time_splits:
        # time based splits:
        for i in np.arange(50, 118, 2):
            train_src = "/home/ubuntu/data/src/phase1.5/2018.03.05/splits/{i}_weekly_train.csv".format(i=i)
            test_src = "/home/ubuntu/data/src/phase1.5/2018.03.05/splits/{i}_weekly_test.csv".format(i=i)
            os.system("""python ccctc_spam.py \
                        --extract \
                        --transform \
                        --train \
                        --test \
                        --train_src {train_src} \
                        --test_src {test_src} \
                        --tag time_split_{i}
                        """.format(train_src=train_src, test_src=test_src, i=i))
            runs += 1

    if do_noise_splits:
        # the noise based splits will come here:
        for noise_split in [60, 65, 70, 75, 80, 85]:
            cfg = os.path.join(paths['CFGDIR'], "ccctc_spam_noise_{noise_split}_train.yaml".format(noise_split=noise_split))
            os.system("""python ccctc_spam.py \
                        --extract \
                        --transform \
                        --train \
                        --test \
                        --cfg {cfg} \
                        --tag noise_split_{noise_split} \
                        --train_src  /home/ubuntu/data/src/phase1.5/2018.03.05/splits/{noise_split}_college_train.csv \
                        --test_src /home/ubuntu/data/src/phase1.5/2018.03.05/splits/{noise_split}_college_test.csv
                        """.format(cfg=cfg, noise_split=noise_split))
            runs += 1

    results_db = os.environ['ML_DB']
    results = pd.read_csv(results_db, parse_dates=["date"])

    # only select results from current run:
    results = results.sort_values("date", ascending=False).iloc[:(runs*2)]
    return results


def run_model_evaluation():

    column_order = ['date',
                    'tag',
                    'logfile',
                    'model',
                    'scope',
                    'AUC',
                    'Accuracy',
                    'Precision',
                    'Recall',
                    'feature0',
                    'feature0_score',
                    'feature1',
                    'feature1_score',
                    'feature2',
                    'feature2_score',
                    'feature3',
                    'feature3_score',
                    'feature4',
                    'feature4_score',
                    'feature5',
                    'feature5_score',
                    'feature6',
                    'feature6_score',
                    'feature7',
                    'feature7_score',
                    'feature8',
                    'feature8_score',
                    'feature9',
                    'feature9_score',
                    'testset',
                    'trainset']

    paths = load_paths()
    results = run_evaluation_splits(paths=paths)
    print(results)
    path = os.path.join(paths["TESTDIR"], "last_experiment_results.csv")
    results.to_csv(path, index=None, columns=column_order)


def run_test():
    args = get_args()
    log, paths, params = setup_env(args)
    transopts = params['load']['transform']
    y_col = transopts['y_column']
    # Generate prediction split
    # generate_prediction_split(paths)
    # Load prediction data
    df = pd.read_csv(os.path.join(paths['SRCDIR'], 'pilot_test_apps_mod.csv'),
                     low_memory=True,
                     true_values=['t', 'T'],
                     false_values=['f', 'F'],
                     dtype=object)
    df.replace(to_replace=['t', 'T'], value=True, inplace=True, method=None)
    df.replace(to_replace=['f', 'F'], value=False, inplace=True, method=None)
    json_object = json_loads_byteified(df.to_json(orient='index'))
    jh = open(os.path.join(paths['SRCDIR'], 'pilot_test_apps_mod2.json'), 'wb')
    # print json
    all_preds_df = pd.DataFrame(columns=['pred_fraud_prob', 'pred_fraud', 'is_fraud'])
    pred_df = pd.DataFrame(columns=['pred_fraud_prob', 'pred_fraud', 'is_fraud'])
    for in_json in json_object:
        jh.write(json.dumps(json_object[in_json]) + '\n')
        pred_df[['pred_fraud_prob', 'pred_fraud', 'app_id']] = predict_fraud(args, paths, params,
                                                                             json_obj=[json_object[in_json]])
        pred_df[y_col] = int(json_object[in_json][y_col])
        all_preds_df = all_preds_df.append(pred_df, ignore_index=True)
        log.info('Prediction: {}'.format(pred_df.to_json(orient='index')))
    jh.close()
    log.debug('Predictions: {}'.format(all_preds_df))
    # Evaluate Predictions
    results = evaluate_predictions(args, paths, params, all_preds_df[y_col].astype('bool'),
                                   all_preds_df['pred_fraud_prob'],
                                   params['test']['probability_threshold'],
                                   scope='spam_test')
    log.info('\n\nPrediction Results: \n{}\n'.format(sf.tabulate_dict(results, multilevel=False, transpose=True)))


if __name__ == '__main__':
    # Run test using the prediction service
    # run_test()
    # Run Evaluation Splits
    run_model_evaluation()
