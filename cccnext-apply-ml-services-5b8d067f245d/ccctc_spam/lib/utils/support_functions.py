################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

#########################################################################################################
#  Description: Collection of support functions that'll be used often
#
#########################################################################################################
import pandas as pd
import numpy as np
import random
import os
import pprint
import logging
import pickle
from logging.config import fileConfig
from itertools import groupby, product
import tabulate
from tabulate import tabulate as tab
from collections import defaultdict
from datetime import datetime
from pytz import timezone

#########################################################################################################
__author__ = 'DataCentric1'
__pass__ = 1
__fail__ = 0

#########################################################################################################

#########################################################################################################
# Set up logging
log = logging.getLogger('info')
#########################################################################################################


# Class to specify color and text formatting for prints
class Color:
    def __init__(self):
        pass

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


#  Returns number of lines in a file in a memory / time efficient way
def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i


# Returns random floating point value within the range specified
def random_float(low, high):
    return random.random()*(high-low) + low


# Returns all elements in the list with format 0.2f
def format_float_0_2f(list_name):
    return "["+", ".join(["%.2f" % x for x in list_name])+"]"


# Load Model Data for a CSV
def load_model_data(data_csv='dummy.csv', low_memory=True):
    """
    Reads a CSV file, Returns a Pandas Data Frame

    :param data_csv:
    :return dataproc:
    """
    if os.path.isfile(data_csv):
        data = pd.read_csv(data_csv, sep=',', low_memory=low_memory)
        return data
    else:
        raise ValueError('Input file %s not available', data_csv)


# Pretty Print Function
class PrettyLog():

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return pprint.pformat(self.obj)


# Encode as UTF-8
def utfenc(s):
    if not s:
        s = ''
    return s.encode('utf-8')


# Error handling to ensure mis-matched options are not used.
def check_options(opts, pickle_file):
    if os.path.isfile(pickle_file):
        popts = pickle.load(open(pickle_file, 'rb'))
        log.info('Loaded pickled options from %s for verification', pickle_file)
        if opts != popts:
            log.error('\tOptions different between current program settings and pickled values...')
            log.error('\n\nProgram Configuration\n%s\n', PrettyLog(opts))
            log.error('\n\nPickled Configuration\n%s\n', PrettyLog(popts))
        else:
            log.info('Options same between current program settings and pickled values...')
    else:
        log.info('\tPickle file (%s) of options does not exist. Tread carefully!', pickle_file)


def n_most_common_elements_list(listname, num_elements=1):
    """
    Returns n Most common items in a list

    :param listname: List on which to operate on
    :param num_elements: Num of common elements to return. Defaults to 1
    :return:
    """

    return max(groupby(sorted(listname)), key=lambda(x, v): (len(list(v)), -listname.index(x)))[0:num_elements]


def tabulate_search_results(d, count=0, condition=None):
    gs_df = pd.DataFrame(d)
    gs_df = gs_df.filter(regex='mean_test_*|rank_*|params')
    if condition:
        col_rank = 'rank_test_score'
        col_mean = 'mean_test_score'
        gs_df = gs_df.sort_values([col_rank, col_mean], ascending=[1, 1])
        gs_df.set_index(col_rank, inplace=True, drop=False)
    if count > 0:
        gs_df = gs_df.head(count)
    return tab(gs_df, headers=gs_df.columns.tolist(), tablefmt='psql')


# Tabulate data for better visualization in logs
def tabulate_list(lst, headers=[]):
    return tab([[v] for v in lst], headers=headers, tablefmt='psql')


# Tabulate data for better visualization in logs
def tabulate_dict(data, multilevel=False, transpose=False, trim_path=False, preserve_whitespace=False):
    d = data.copy()
    if preserve_whitespace:
        tabulate.PRESERVE_WHITESPACE = True
    if multilevel:
        df = pd.DataFrame(d)
        if transpose:
            return tab(df.T, headers=df.T.columns.tolist(), tablefmt='psql')
        else:
            return tab(df, headers=df.columns.tolist(), tablefmt='psql')
    else:
        l = list()
        if transpose:
            for k, v in d.iteritems():
                if trim_path:
                    v = os.path.basename(v)
                l.append([k, v])
            return tab(l, headers=['Attribute', 'Value'], tablefmt='psql')
        else:
            for k, v in d.iteritems():
                if not isinstance(v, (list, tuple)):
                    if trim_path:
                        v = os.path.basename(v)
                    d[k] = [v]
            return tab(d, headers='keys', tablefmt='psql')


# Create a custom Grid Search List
def generate_custom_grid(d):
    keys = sorted(d)
    return [dict(zip(keys, prod)) for prod in product(*(d[key] for key in keys))]


# Setup to work-around pickling issue
def dd_dict():
    return dict()

def ip_to_decimal(x):
    split = x.split('.')
    total = 0
    for i in range(4):
        total += int(split[-(i+1)])*(256**(i))
    return total

# Map IP address to country
def ip_to_country_helper(num_array, country_dict):
    def ip_to_country(x):
        try:
            arr_len = len(num_array)
            x = ip_to_decimal(x)
            end = arr_len
            start = 0
            idx = start + (end - start) // 2
            start_n = num_array[idx]
            end_n = num_array[idx+1]
            while not ((x >= start_n) and (x < end_n)):
                if x < start_n:
                    end = idx
                else:
                    start = idx
                idx = start + (end - start) // 2
                start_n = num_array[idx]
                if idx+1 >= arr_len:
                    return '-'
                end_n = num_array[idx+1]
            country = country_dict[start_n]
            return country
        except:
            return 'ZZ'
    return ip_to_country


# Get current timestamp
def get_timestamp(tzinfo='US/Pacific'):
    tz = timezone(tzinfo)
    return datetime.now(tz).isoformat()


# Append results to DB
def db_append_results(args, paths, params, model_class=None, feature_importance=None, results=None, scope='train'):
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
    db_results = results.copy()
    db_results['tag'] = args.tag
    db_results['date'] = str(pd.Timestamp.now())
    db_results['trainset'] = paths['SRCFILE']
    if 'TEST_SRC' in paths:
        db_results['testset'] = paths['TEST_SRC']
    else:
        db_results['testset'] = paths['SRCFILE']
    db_results['logfile'] = params['LOGNAME']
    db_results['model'] = model_class
    db_results['scope'] = scope
    if scope == 'train':
        i = 0
        for k, v in feature_importance.items():
            db_results['feature' + str(i)] = k
            db_results['feature' + str(i) + '_score'] = v
            i += 1
    else:
        for i in range(10):
            db_results['feature' + str(i)] = np.nan
            db_results['feature' + str(i) + '_score'] = np.nan
    db_results = pd.DataFrame(db_results).T
    log.info('Test Results Summary (Recorded to DB):\n{}'.format(tab(db_results, headers=db_results.columns.tolist(), tablefmt='psql')))
    if os.path.isfile(paths['ML_DB']):
        previous_results = pd.read_csv(paths['ML_DB'])
        db_results = pd.concat([previous_results, db_results], axis=0)
    db_results.to_csv(paths["ML_DB"], columns=column_order, index=None)
    return
