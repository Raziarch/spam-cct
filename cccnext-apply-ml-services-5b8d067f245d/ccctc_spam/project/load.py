################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################


# Standard Packages
import os
import logging
import pickle
import re
import pandas as pd
import numpy as np
from datetime import datetime, date
import csv
import yaml

# ML Packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import local packages

from lib.utils import support_functions as sf
from lib.utils import encoders as enc
from lib.utils import modeling_tools as mt
from lib.utils import report

#########################################################################################################
# Set up logging
log = logging.getLogger('info')
#########################################################################################################


# Format string to timex
def s2time(s):
    return datetime.strptime(str(s), '%Y-%m-%d %H:%M')


# Update ETL directories
def update_etl_paths(paths, params):
    params['config']['raw_pickle_file'] = os.path.join(paths['EOUTDIR'],
                                                       params['config']['raw_pickle_file'])
    params['config']['transform_pickle_file'] = os.path.join(paths['EOUTDIR'],
                                                             params['config']['transform_pickle_file'])
    params['config']['extract_cfgopts_file'] = os.path.join(paths['EOUTDIR'],
                                                            params['config']['prefix'] +
                                                            '_extract_cfgopts.pkl')
    params['config']['transform_cfgopts_file'] = os.path.join(paths['EOUTDIR'],
                                                              params['config']['prefix'] +
                                                              '_transform_cfgopts.pkl')
    return params


# Trim fields that are not useful
def trim_data(df, transopts, extropts, scope='train'):
    log.info('Function Start')
    log.info('Trimming data set to optimize ML inputs.....')
    log.info('\tShape of source dataframe: {}'.format(df.shape))
    # Collate columns to delete
    if transopts['trim']['trim_filter_regex']:
        del_cols = []
        for ex in transopts['trim']['trim_filter_regex']:
            regex = re.compile(ex)
            # Delete columns by regular expression
            del_cols = del_cols + [col for col in df.columns.tolist() if re.search(regex, col)]
        # Delete columns explicitly
        del_cols = list(set(del_cols) | set(transopts['trim']['trim_features']))
        # Remove Exceptions
        del_cols = list(set(del_cols) - (set(transopts['trim']['trim_filter_exceptions']) & set(del_cols)))
        log.debug('\n\nList of columns that will be removed: \n{}\n'.
                 format(sf.tabulate_list(lst=sorted(del_cols), headers=['Removed Columns'])))
        log.info('\tNumber of columns removed: {}'.format(len(del_cols)))
        df.drop(del_cols, axis=1, inplace=True, errors='ignore')
        log.info('\tShape of reduced dataframe: {}'.format(df.shape))
    if scope == "train" and transopts['cardinality_thresh']:
        log.info('Limiting cardinality for features:')
        numeric_features = df.select_dtypes(['int', 'float', 'bool']).columns.tolist()
        categorical_features = df.columns.difference(numeric_features).tolist()
        limit_cardinality(df, categorical_features, min_thresh=transopts['cardinality_thresh'])
    # Return the columns in the dataframe that are numeric (or)
    # have lower than the unique_bound of unique values
    if transopts['unique_bound']:
        uniq_counts = (df.select_dtypes(['object'])
                       .apply(lambda x: len(x.unique()))
                       .sort_values())
        good_categories = uniq_counts[uniq_counts < transopts['unique_bound']].index.tolist()
        numeric = df.select_dtypes(['int', 'float', 'bool']).columns.tolist()
        log.info('Function End')
        return df[good_categories + numeric]
    else:
        log.info('Function End')
        return df


def limit_cardinality(df, cols, min_thresh=10, lowercase=True):
    """
    If a category is occurring less than min_thresh times replace it
    with 'Other'.
    """
    for col in cols:
        log.debug('Checking if cardinality limiting is needed for column: {}; Data Type: {}'.format(col, df[col].dtype))
        df[col] = df[col].str.lower()
        counts = df[col].value_counts()
        low_values = list(counts[counts <= min_thresh].index)
        if low_values:
            log.info('Applied Cardinality Threshold for {}....'.format(col))
        df.loc[df[col].isin(low_values), col] = "Other"


# Add new features to support ML
# TODO: Add email feature that removes all special characters (For Post PII)
def augment_data(df, paths, transopts, scope='train'):
    log.info('Function Start')
    # Removing white spaces in column name
    df.columns = df.columns.str.replace('\s+', '')
    log.info('\tAugmenting data, feature engineering.....')
    log.info('\tAdding app_duration feature...')
    # Creating application duration column
    df["tstmp_create"] = pd.to_datetime(df["tstmp_create"])
    df["tstmp_submit"] = pd.to_datetime(df["tstmp_submit"])
    df["app_duration"] = (df["tstmp_submit"] - df["tstmp_create"]) / np.timedelta64(1, 's')
    # log.info('\tAdding time_stamp features..')
    df['create_wday'] = df['tstmp_create'].dt.weekday
    df['create_hour'] = df['tstmp_create'].dt.hour
    df['create_4hrblock'] = 4 * (df['create_hour'] // 4)
    df['submit_wday'] = df['tstmp_submit'].dt.weekday
    df['submit_hour'] = df['tstmp_submit'].dt.hour
    df['submit_4hrblock'] = 4 * (df['submit_hour'] // 4)
    # Development dataset doesn't have PII data. email column not available
    # Only email_domain column available. Extending code to support email column in production
    log.info('\tAdding email domain related features...')
    if 'email' in df.columns.tolist() and not transopts['anonymized_email']:
        df['email'] = df['email'].str.lower()
        df['email_stripped'] = df['email'].str.replace('[^\w]', '')
        df['email_domain'] = df['email'].str.split('@', expand=True)[1]
        # Generating short domain name
        df['short_domain'] = df['email_domain'].str.split('.', expand=True)[1]
        
        if transopts['blacklist_feature']:
            domain_feature_blacklist_path = os.path.join(paths['FINALDIR'], 'domain_feature_blacklist.txt')
            domain_feature_blacklist = set(pd.read_csv(domain_feature_blacklist_path, header=None)[0].str.lower())
            is_domain_feature_blacklisted = df['email'].apply(lambda x: x.split('.')[-1].lower() in domain_feature_blacklist)
            df['blacklisted_domain'] = is_domain_feature_blacklisted
            
    elif 'email' in df.columns.tolist() and transopts['anonymized_email']:
        df['email_domain'] = df['email'].str.lower()
        # Generating short domain name
        df['short_domain'] = df['email_domain'].str.split('.', expand=True)[1]
    # Splitting out race/ethnicity, re-assigning to Boolean values
    if not transopts['augment']['exclude_race_gender']:
        log.info('\tEnumerating race_ethnic encoded feature.....')
        df[['is_HisL', 'is_Mex', 'is_CeAm', 'is_SoAm', 'is_HisO', 'is_AsInd', 'is_AsChi', 'is_AsJap', 'is_AsKor',
            'is_AsLao', 'is_AsCam', 'is_AsVie', 'is_Fil', 'is_AsO', 'is_AA', 'is_AmInd', 'is_PacIG', 'is_PacIH',
            'is_PacIS', 'is_PacIO', 'is_Whi']] = df['race_ethnic'].apply(lambda x: pd.Series(list(x)))
        df.replace({'Y': True, 'N': False}, inplace=True)
    # Splitting date columns
    if transopts['augment']['expand_dates']:
        log.info('\tExpanding date columns....')
        # date_cols = [col for col in df.columns.tolist() if col.endswith('_date')]
        date_cols = [c for c in df.columns.tolist() if ('_date' in c) or ('_year' in c)]
        for col in date_cols:
            # df[col] = pd.to_datetime(df[col], format='YYYY-MM-DD', errors='coerce')
            df[col] = pd.to_datetime(df[col], errors='coerce')
            year_col = col + '_y'
            month_col = col + '_m'
            day_col = col + '_d'
            df[year_col] = df[col].dt.year
            df[month_col] = df[col].dt.month
            df[day_col] = df[col].dt.day
    # Fix Zip Codes: Remove Zip Extension, Adjust Alphabetical Zips, Remove Spaces
    for col in ['postalcode', 'perm_postalcode']:
        log.info('Cleaning up {}....'.format(col))
        df[col] = (df[col].astype(str).str.split('[.-]', expand=True)[0].str.replace('[a-zA-Z -,/:]', '0'))
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace('', '0')
        df[col] = df[col].astype(float)
    # Fraud Tag Override Function
    if transopts['pii']['enable']:
        log.info('Adding pii features...')
        pii_cols = transopts['pii']['pii_na_transform']
        pii_na_cols = list(map(lambda x: x + '_isna', pii_cols))
        df_na = df[pii_cols].copy().isna()
        df_na.columns = pii_na_cols
        df = pd.concat([df, df_na], axis=1)
        df['num_pii_na'] = df_na.sum(axis=1)
        del df_na
        df['birthdate'] = df['birthdate'].map(lambda x: pd.to_datetime(x) if int(x.split('-')[0]) <= date.today().year else np.nan, na_action='ignore')
        df['birth_day'] = df['birthdate'].map(lambda x: x.day, na_action='ignore')
        df['birth_month'] = df['birthdate'].map(lambda x: x.month, na_action='ignore')
        df['birth_year'] = df['birthdate'].map(lambda x: x.year, na_action='ignore')
        df['application_age'] = df['tstmp_submit'] - df['birthdate']
        df['application_age'] = df['application_age'].map(lambda x: x.days // 365 if not pd.isna(x) else np.nan, na_action='ignore')
        df['hs_comp_age'] = df['hs_comp_date'] - df['birthdate']
        df['hs_comp_age'] = df['hs_comp_age'].map(lambda x: x.days // 365 if not pd.isna(x) else np.nan, na_action='ignore')
        df['tstmp_submit_hour'] = df['tstmp_submit'].map(lambda x: x.hour, na_action='ignore')
        df['tstmp_submit_weekday'] = df['tstmp_submit'].map(lambda x: x.weekday(), na_action='ignore')
        df['firstname_length'] = df['firstname'].map(lambda x: len(x) if type(x) != bool else 1, na_action='ignore')
        df['middlename_length'] = df['middlename'].map(lambda x: len(x) if type(x) != bool else 1, na_action='ignore')
        df['lastname_length'] = df['lastname'].map(lambda x: len(x) if type(x) != bool else 1, na_action='ignore')
        df['area_code'] = df['mainphone'].map(lambda x: x.split(' ')[0], na_action='ignore')
        df['in_CA'] = df['state'].map(lambda x: x == 'CA', na_action='ignore')
        df['ip_address'] = df['ip_address'].map(lambda x: '000.000.000.000' if x == 'unknown' else x, na_action='ignore')
        df_tmp = df['ip_address'].str.split('.', expand=True)
        for i in range(4):
            df['ip' + str(i)] = df_tmp[i]
        country_idx_array = pickle.load(open(os.path.join(paths['FINALDIR'], 'ip_country_idx.pk'), 'rb'))
        country_dict = pickle.load(open(os.path.join(paths['FINALDIR'], 'ip_country_dict.pk'), 'rb'))
        area_dict = pickle.load(open(os.path.join(paths['FINALDIR'], 'area_code_state_dict.pk'), 'rb'))
        area_code_to_state_fn = area_code_state_helper(area_dict)
        df['area_code_state'] = df['area_code'].apply(area_code_to_state_fn)
        ip_to_country_fn = sf.ip_to_country_helper(country_idx_array, country_dict)
        df['ip_country'] = df['ip_address'].apply(ip_to_country_fn)
        df['ip_US'] = df['ip_country'].apply(lambda x: x == 'US')
    if transopts['pii']['enable']:
        create_rank(df, paths, transopts, scope)
        log.info('Completed pii features.')
    if transopts['tag_override'] and scope == 'train':
        df = tag_override(df, transopts)
    # Renaming the output column
    df.rename(columns={transopts['y_column']: 'y'}, inplace=True)
    log.info('Function End')
    return df

def area_code_state_helper(area_dict):
    def area_code_to_state(x):
        if type(x) != str:
            return 'na'
        if len(x) != 5:
            return 'na'
        x = x[1:-1]
        if x in area_dict:
            state = area_dict[x]
            if state == 'CA':
                return 'CA'
            else:
                return 'non-CA'
        else:
            return 'unknown'
    return area_code_to_state

def create_rank(df, paths, transopts, scope):
    log.info('Creating rank transformed features....')
    features = transopts['pii']['rank_features']
    if scope == 'train':
        rank_dict = {}
        for feature in features:
            ranking = list(df[feature].value_counts().index)
            thresh = int(len(ranking)*transopts['pii']['thresh'])
            ranking = ranking[:thresh]
            rank_dict[feature] = ranking
        pickle.dump(rank_dict, open(os.path.join(paths['FINALDIR'], 'feature_ranks.pkl'), 'wb'))
    else:
        rank_dict = pickle.load(open(os.path.join(paths['FINALDIR'], 'feature_ranks.pkl'), 'rb'))
    for feature in features:
        ranking = rank_dict[feature]
        df[feature + '_rank'] = df[feature].map(lambda x: ranking.index(x) if x in ranking else len(ranking), na_action='ignore')

# Extract data from raw data source
def extract_source(args, paths, params, scope='train'):
    config = params['config']
    extropts = params['load']['extract']
    transopts = params['load']['transform']
    # Extract Code
    if scope == 'train':
        csvfile = paths['SRCFILE']
    if scope == 'test':
        csvfile = paths['TEST_SRC']
    log.info('Input CSV file: {}'.format(csvfile))
    if extropts['force_datatypes']:
        # Capture data types for input CSV file from YAML
        feats = extropts['base_features']
        dtypes = {k: v for k, v in feats.items() if 'date' not in v}
        # Pandas does not support missing values for integer dtypes
        # As a result, we change int to float
        for key, value in dtypes.items():
            if value == "int":
                if key != transopts['id_column']:
                    dtypes[key] = "float"
        df = pd.read_csv(csvfile,
                         low_memory=True,
                         dtype=dtypes,
                         true_values=['t', 'T'],
                         false_values=['f', 'F'])
    else:
        df = pd.read_csv(csvfile,
                         low_memory=True,
                         true_values=['t', 'T'],
                         false_values=['f', 'F'])
    df = df.replace(to_replace=[r'^None$', r'^ *$'], value=[np.nan, np.nan], regex=True)
    if scope == 'train':
        log.info('run_metrics: raw_input_data_rows - {}'.format(df.shape[0]))
        log.info('run_metrics: raw_input_data_cols - {}'.format(df.shape[1]))
        fraud_score = df['fraud_score']
    else:
        fraud_score = None
    # Dropping features not present in the base features list
    if extropts['base_features']:
        base_features = extropts['base_features'].keys()
        log.info('Dropping {} features not present in the base features list....'
                 .format(len(df.columns.difference(base_features))))
        df.drop(df.columns.difference(base_features), axis=1, inplace=True)
    # Dump CSV Output (Used only in development)
    if extropts['output_formats'] is not None and \
                    'csv' in extropts['output_formats'] and \
            not params['production']['freeze']:
        extract_csv_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_' + scope + '_extracted.csv')
        log.info('\tWriting out extracted CSV ({}) for inspection....'.format(extract_csv_file))
        df.to_csv(extract_csv_file, index=False)
    # Pickle Data for Quick Load (Used only in development)
    if not params['production']['freeze']:
        # Pickle Raw Output
        extract_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_' + scope + '_extracted.pkl')
        pickle.dump(df, open(extract_pkl_file, 'wb'))
        log.info('\tPickled raw source data to %s', extract_pkl_file)
        # Pickle Config Options
        pickle.dump(config, open(config['extract_cfgopts_file'], 'wb'))
        log.info('\tPickled config to %s for future reference', config['extract_cfgopts_file'])
    return df, fraud_score


# Override Fraud Tag
# Used to add specific filters
# Overtime, this are not necessary as the machine learning model will have learnt it
def tag_override(df, transopts):
    # If Application Duration < app_duration_threshold
    log.info('Number of Tagged Fraud before Retag: {}'.format(df[df[transopts['y_column']] == True].shape[0]))
    df.loc[df['app_duration'] < transopts['app_duration_threshold'], transopts['y_column']] = True
    log.info('Number of Tagged Fraud after Retag: {}'.format(df[df[transopts['y_column']] == True].shape[0]))
    return df


# Transform data after extraction
def transform_source(args, paths, params, df, scope='train'):
    if len(df) == 1:
        log.info('Processing app_id: {} ...'.format(df['app_id'].iloc[0]))
    override_yaml_path = os.path.join(paths['FINALDIR'], 'override.yaml')
    override_opts = yaml.load(open(override_yaml_path, 'r'))

    config = params['config']
    transopts = params['load']['transform']
    trainopts = params['train']
    extropts = params['load']['extract']
    
    out_csv_file = ''
    if scope == 'train':
        log.info('run_metrics: total_tagged_fraud - {}'.format(df[transopts['y_column']].value_counts()[True]))
    # During prediction, ensure only base features seen in training are used.
    if params['production']['freeze'] and scope == 'predict':
        base_features = extropts['base_features'].keys()
        # Remove y from the base features list in prediction mode
        base_features = list(set(base_features) - set([transopts['y_column']]))
        log.info('Dropping {} features not present in the base features list....'
                 .format(len(df.columns.difference(base_features))))
        df.drop(df.columns.difference(base_features), axis=1, inplace=True)
    # Set index to submit timestamp
    df['ts'] = pd.to_datetime(df[transopts['ts_column']])
    # Create Features
    log.info('\tAugmenting features.....')
    df = augment_data(df, paths, transopts, scope=scope)
    # Add rule-based prediction   
    log.info('\tApplying rule based prediction....')
    y_rule = rule_based_prediction(df, paths, transopts, override_opts, scope)
    # Override training labels using rule-based model
    df = rule_based_override(df, y_rule, paths, transopts, override_opts, scope)
    # Trim unwanted features
    log.info('\tTrimming unwanted features.....')
    df = trim_data(df, transopts, extropts, scope=scope)
    # log.info('\tAdding missing values features')
    agg_cols = list(set(df.columns) - set(['y']))
    df["zeros"] = (df[agg_cols] == 0).sum(1)
    df["missing_values"] = (df[agg_cols].isnull()).sum(1)
    # Encode dataframe
    log.info('\tEncoding data.....')
    enc_df = enc.encode(df, paths, params, scope=scope)
    # Dump CSV Output (Used only in development)
    if transopts['output_formats'] is not None and \
            'csv' in transopts['output_formats'] and \
            not params['production']['freeze']:
        transform_csv_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_' + scope + '_transformed.csv')
        log.info('\tWriting out transformed CSV ({}) for inspection....'.format(transform_csv_file))
        enc_df.to_csv(transform_csv_file, index=False)
    return enc_df, df, y_rule

def rule_based_override(df, y_rule, paths, transopts, override_opts, scope):
    if not override_opts['override']:
        return df
    if override_opts['override_tagging'] and scope == 'train':
        log.info('\tOverriding training data tagging...')
        df['y'] = (df['y'] | y_rule)
    return df

def rule_based_prediction(df, paths, transopts, override_opts, scope):
    if not override_opts['override']:
        log.info('\tOverride set to False - No rule based prediction applied.')
        return None
    rule_list = []
    if override_opts['time_limit']['active']:
        log.info('\tSetting rule-based time limit override....')
        below_thresh = np.array(df['app_duration'] <= override_opts['time_limit']['threshold'])
        rule_list.append(below_thresh)
    if override_opts['email_blacklist']['active'] and 'email' in df.columns.tolist() and not transopts['anonymized_email']:
        log.info('\tSetting email rule-based blacklist....')
        email_handle = df['email'].str.split('@', expand=True)[0]
        if override_opts['email_blacklist']['period_blacklist']:
            log.info('\tBlacklisting addresses containing a period....')
            period_blacklist_path = os.path.join(paths['FINALDIR'], 'period_blacklist.txt')
            period_blacklist = set(pd.read_csv(period_blacklist_path, header=None)[0])
            has_period = np.array(email_handle.apply(lambda x: '.' in x))
            is_period_domain = np.array(df['email_domain'].isin(period_blacklist))
            period_blacklist = has_period & is_period_domain
            rule_list.append(period_blacklist)
        if override_opts['email_blacklist']['address_blacklist']:
            log.info('\tBlacklisting email addresses in address blacklist....')
            address_blacklist_path = os.path.join(paths['FINALDIR'], 'address_blacklist.txt')
            address_blacklist = set(pd.read_csv(address_blacklist_path, header=None)[0].str.lower())
            is_blacklisted = np.array(df['email_domain'].isin(address_blacklist))
            rule_list.append(is_blacklisted)
        if override_opts['email_blacklist']['top_domain_blacklist']:
            log.info('\tBlacklisting top level domains in top domain blacklist....')
            top_domain_blacklist_path = os.path.join(paths['FINALDIR'], 'top_domain_blacklist.txt')
            top_domain_blacklist = set(pd.read_csv(top_domain_blacklist_path, header=None)[0].str.lower())
            is_top_domain_blacklisted = df['email'].apply(lambda x: x.split('.')[-1].lower() in top_domain_blacklist)
            rule_list.append(is_top_domain_blacklisted)
    if len(rule_list) == 0:
        return None
    else:
        rule_decision = rule_list[0]
        if len(rule_list) > 1:
            for i in range(1, len(rule_list)):
                rule_decision = rule_decision | rule_list[i]
        rule_decision_int = rule_decision.astype(int)
        return rule_decision_int


# Load previously extracted file
def load_extracted_source(args, paths, params, scope='train'):
    config = params['config']
    log.info('\n\nBase Configuration\n{}\n'.format(sf.tabulate_dict(config, transpose=True, trim_path=True)))
    extract_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_' + scope + '_extracted.pkl')
    log.info('Loading pickled raw data from %s', extract_pkl_file)
    df = pickle.load(open(extract_pkl_file, 'rb'))
    return df


def pickle_ml_data(args, paths, params, train_df, train_unenc_df, test_df, test_unenc_df):
    log.info('Function Start')
    log.info('Pickling train/test data for future use...')
    config = params['config']
    train_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_train_transformed.pkl')
    pickle.dump(train_df, open(train_df_pkl_file, 'wb'))
    train_unenc_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_train_unenc_transformed.pkl')
    pickle.dump(train_unenc_df, open(train_unenc_df_pkl_file, 'wb'))
    test_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_test_transformed.pkl')
    pickle.dump(test_df, open(test_df_pkl_file, 'wb'))
    test_unenc_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_test_unenc_transformed.pkl')
    pickle.dump(test_unenc_df, open(test_unenc_df_pkl_file, 'wb'))
    log.info('Function End')
    return


def load_pickled_ml_data(args, paths, params):
    log.info('Function Start')
    log.info('Loading pickled train/test data....')
    config = params['config']
    train_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_train_transformed.pkl')
    train_df = pickle.load(open(train_df_pkl_file, 'rb'))
    train_unenc_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_train_unenc_transformed.pkl')
    train_unenc_df = pickle.load(open(train_unenc_df_pkl_file, 'rb'))
    test_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_test_transformed.pkl')
    test_df = pickle.load(open(test_df_pkl_file, 'rb'))
    test_unenc_df_pkl_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_test_unenc_transformed.pkl')
    test_unenc_df = pickle.load(open(test_unenc_df_pkl_file, 'rb'))
    log.info('Function End')
    return train_df, train_unenc_df, test_df, test_unenc_df


def etl_data(args, paths, params):
    log.info('Function Start')
    params = update_etl_paths(paths, params)
    transopts = params['load']['transform']
    trainopts = params['train']
    testopts = params['test']
    reportopts = params['report']
    # If a specific test source is provided
    if args.test_src:
        # If extract options is selected
        if args.extract:
            train_df, train_score = extract_source(args, paths, params, scope='train')
            test_df, test_score = extract_source(args, paths, params, scope='test')
        # Load pickled extract files only if transform is being done later.
        elif args.transform:
            train_df = load_extracted_source(args, paths, params, scope='train')
            test_df = load_extracted_source(args, paths, params, scope='test')
        if reportopts['enable']:
            log.info('Generating report...')
            report.make_report(train_df, train_score, reportopts, paths, log)
        # If transform option is selected
        if args.transform:
            train_df, train_unenc_df, train_y_rule = transform_source(args, paths, params, train_df, scope='train')
            test_df, test_unenc_df, test_y_rule = transform_source(args, paths, params, test_df, scope='test')
            if not params['production']['freeze']:
                pickle_ml_data(args, paths, params, train_df, train_unenc_df, test_df, test_unenc_df)
        # If transform option is not selected, pickled files are loaded
        else:
            train_df, train_unenc_df, test_df, test_unenc_df = load_pickled_ml_data(args, paths, params)
    # If no test source is provided. Splitting done internally
    else:
        # If extract option is selected
        if args.extract:
            df, score = extract_source(args, paths, params, scope='train')
        # Load pickled extract file only if transform is being done
        elif args.transform:
            df = load_extracted_source(args, paths, params, scope='train')
        # if transform options is selected
        if reportopts['enable']:
            log.info('Generating report...')
            report.make_report(df, score, reportopts, paths, log)
        if args.transform:
            df, unenc_df, y_rule = transform_source(args, paths, params, df, scope='train')
            split = mt.split_data(args, paths, params, df, scope='train_test')
            train_index, test_index = split[0]
            if testopts['split_method'] == 'split_by_time' and args.test:
                df.index = df['ts']
                unenc_df.index = df.index
                train_df = df.loc[train_index]
                test_df = df.loc[test_index]
                train_unenc_df = unenc_df.loc[train_index]
                test_unenc_df = unenc_df.loc[test_index]
            else:
                train_df = df.loc[train_index]
                test_df = df.loc[test_index]
                train_unenc_df = unenc_df.loc[train_index]
                test_unenc_df = unenc_df.loc[test_index]
            if params['load']['transform']['debug'] and not params['production']['freeze']:
                log.debug('Dumping post split train, test dataframes....')
                train_df.to_csv(os.path.join(paths['EOUTDIR'], 'train_split.csv'), index=False)
                test_df.to_csv(os.path.join(paths['EOUTDIR'], 'test_split.csv'), index=False)
            if not params['production']['freeze']:
                pickle_ml_data(args, paths, params, train_df, train_unenc_df, test_df, test_unenc_df)
        # if transform option is not selected, load pickled files
        else:
            train_df, train_unenc_df, test_df, test_unenc_df = load_pickled_ml_data(args, paths, params)
        test_y_rule = None 
    return train_df, train_unenc_df, test_df, test_unenc_df, test_y_rule


# Load Data
def load_predict_source(json_obj, paths, params):
    log.info('Function Start')
    params = update_etl_paths(paths, params)
    config = params['config']
    transopts = params['load']['transform']
    predopts = params['predict']
    # Convert JSON string to a dataframe
    df = pd.DataFrame(json_obj)
    df.fillna(value=np.nan, inplace=True)
    # Create Features
    df = augment_data(df, transopts)
    # Trim unwanted features
    df = trim_data(df, transopts)
    # Encode dataframe
    log.info('\tEncoding data.....')
    enc_df = enc.encode(df, paths, params, scope='predict')
    # Dump CSV Output (Used only in development)
    if predopts['debug']:
        transform_csv_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_predict_transformed.csv')
        log.info('\tWriting out transformed CSV ({}) for inspection....'.format(transform_csv_file))
        enc_df.to_csv(transform_csv_file, index=False)
    log.info('Function End')
    return enc_df


# Generate X, y data from source data frame
def prepare_ml_data(args, paths, params, df, scope='train'):
    log.info('Function Start')
    transopts = params['load']['transform']
    config = params['config']

    df.reset_index(drop=True, inplace=True)
    col_names = df.columns.tolist()

    # Sample Data/Columns
    log.debug(sf.Color.BOLD + sf.Color.GREEN + "Column names:" + sf.Color.END)
    log.debug(col_names)
    log.debug(df.head(6))

    log.info('Splitting X; Converting to numpy arrays.....')
    # Remove ID column from ML source

    if 'y' in df:
        to_drop = ['y']
        y = np.array(df['y'])
        in_df = df.drop(to_drop, axis=1)
        log.debug('\tUnique target labels: {}'.format(np.unique(y)))
    else:
        in_df = df

    if transopts['id_column'] in in_df:
        in_df.drop([transopts['id_column']], axis=1, inplace=True)

    # If split was time-based, remove 'ts' column
    if 'ts' in in_df.columns.tolist():
        in_df.drop(['ts'], axis=1, inplace=True)

    # If split was time-based, remove 'ts' column
    if 'idx' in in_df.columns.tolist():
        in_df.drop(['idx'], axis=1, inplace=True)

    # keep track of feature name ordering:
    if params['production']['freeze']:
        feature_names_file = os.path.join(paths['FINALDIR'], config['extension'] + '_feature_names.pkl')
    else:
        feature_names_file = os.path.join(paths['TRAINDIR'], config['extension'] + '_feature_names.pkl')

    if scope == 'train':
        feature_names = in_df.columns.tolist()
        log.info(sf.Color.BOLD + sf.Color.GREEN + "Feature Names:" + sf.Color.END)
        log.info(feature_names)
        log.info('Saving Feature Names:')
        pickle.dump(feature_names, open(feature_names_file, 'wb'))
    else:
        log.info('Loading Feature Names:')
        feature_names = pickle.load(open(feature_names_file, 'rb'))

    # Add noise to selected features
    if (scope == 'train' or scope == 'test') and params[scope]['feature_noise'] and not params['production']['freeze']:
        log.info("Adding noise to {scope}!".format(scope=scope))
        in_df = mt.generate_input_noise(in_df,
                                        noise_level=params[scope]["noise_level"],
                                        data_perc=params[scope]["noise_percentage"],
                                        feats=params[scope]['noise_features'],
                                        debug=False,
                                        pickle_file_root=os.path.join(paths['EOUTDIR'], 'df_input_noise'))

    in_df.fillna(transopts['fillna_numeric'], inplace=True)
    X = in_df[feature_names].as_matrix().astype(np.float)

    if transopts['feature_scaling']:
        log.info('\tRunning Feature Scaling....')
        if params['production']['freeze']:
            scaler_pickle_file = os.path.join(paths['FINALDIR'], config['extension'] + '_feature_scaler.pkl')
        else:
            scaler_pickle_file = os.path.join(paths['TRAINDIR'], config['extension'] + '_feature_scaler.pkl')

        if scope != 'train':
            scaler = pickle.load(open(scaler_pickle_file, 'rb'))
        else:
            # Feature Scaling and Normalization
            scaler = StandardScaler()
        X = scaler.fit_transform(X)
        if scope == 'train':
            pickle.dump(scaler, open(scaler_pickle_file, 'wb'))

    log.info('\tFeature Space holds %d Observations and %d Features' % X.shape)

    # Add noise to the data - Only for validation
    if params['evaluate']['noise_ratio']:
        log.info(sf.Color.BOLD + sf.Color.GREEN +
                 'Beware: Introducing Noise! Ratio: {}'.format(params['evaluate']['noise_ratio']) + sf.Color.END)
        log.info(sf.Color.BOLD + sf.Color.GREEN +
                 'Only used for validation. Data pickles need to be purged before real runs' + sf.Color.END)
        y = mt.generate_output_noise(y, params['evaluate']['noise_ratio'])

    log.info('Function End')
    if scope == 'train' or scope == 'test':
        return X, y
    else:
        return X
