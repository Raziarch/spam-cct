################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

#########################################################################################################
#  Description: Encode source data before ML work
# 1. Use ce.OrdinalEncoder() (with RandomForest or Xgboost)
#    Record feature importance and model performance (function ordinal below)
# 2. Use Using ce.HashingEncoder(n_components = 16 - 256)
#    The more components the less likely the chance of collisions but the larger the space
# 3. Try Increasing n_components and record performance increase
# 4. Use ce.OneHotEncoder on all categorical variables if memory/time permits.
# 5. If Step 4 is too slow (either at preprocessing time or modelling time)
#    Use numerical features plus ce.OneHotEncoder on categorical features found important in step 1.
# 6. Use the DataFrame Mapper script below for trying specific encoding for each column
#    Use One Hot Encoding on features with less than 10-20 levels
#    Use OrdinalEncoder or LeaveOneOutEncoder on features that have high cardinality
#########################################################################################################

import os
import sys
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from category_encoders import HashingEncoder, OneHotEncoder, LeaveOneOutEncoder, OrdinalEncoder
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn_pandas import DataFrameMapper

# TODO: look a little more into why you get this warning
warnings.simplefilter(action='ignore', category=FutureWarning)

#########################################################################################################
# Set up logging
log = logging.getLogger('info')
#########################################################################################################


def fill_missing(df):
    """Replace the Missing Values in the categorical columns of a dataframe
    by Unknown."""
    obj_cols = list(df.select_dtypes(["object"]).columns)
    df[obj_cols] = df[obj_cols].fillna("Unknown")
    return df


def fit_encoder(df, encoder_dict):
    """Fit the encodings based on a given encoding dictionary.
    """
    encode_list = [(col, enc) for col, enc in encoder_dict.items()]
    mapper = DataFrameMapper(encode_list, df_out=True)
    mapper.fit(df)
    return mapper


def transform_encoder(df, mapper):
    """Return the transformed dataframe based on a given mapper.
    """
    return mapper.transform(df)


def fit_transform_encoder(df, transform_opts={}, out_enc='', out_csv=''):
    log.info('Function Start')
    if transform_opts['set_ordinal_default']:
        enc = OrdinalEncoder()
        encoded_df = enc.fit_transform(df)

    if out_enc:
        pickle.dump(enc, open(out_enc, 'wb'))

    if out_csv:
        # Combine encoded and regular dataframe before dumping CSV
        pd.concat([encoded_df, df], axis=1, join_axes=[encoded_df.index]).to_csv(out_csv, index=False)
        # encoded_df.to_csv(out_csv, index=False)
    log.info('Function End')
    return encoded_df


# Custom Ordinal Encoder - Fit Function
def fit(df, cat_feats):
    encoder = {}
    for column in cat_feats:
        _, indexer = pd.factorize(df[column])
        encoder[str(column)] = indexer
    return encoder


# Custom Ordinal Encoder - Transform Function
def transform(df, cat_feats, encoder):
    # Here we could replace the columns in the original dataframe
    # but it significantly slower (like 10x slower)
    df_enc = pd.DataFrame()
    for column in cat_feats:
        df[column] = df[column].astype(encoder[column].dtype)
        indexer = encoder[str(column)]
        # Get_indexer is basically an inverse transform
        # It returns the corresponding integer for the given label
        # If the category is not found -1 is returned
        df_enc[column] = indexer.get_indexer(df[column])
    return df_enc


# Custom Ordinal Encoder - Fit/Transform
def fit_transform(df, cat_feats):
    encoder = fit(df, cat_feats)
    return transform(df, cat_feats, encoder), encoder


# Encode wrapper for custom label encoder
def encode(df, paths, params, scope='train'):
    log.info('Function Start')
    config = params['config']
    transopts = params['load']['transform']
    if params['production']['freeze']:
        base_path = paths['FINALDIR']
    else:
        base_path = paths['TRAINDIR']
    out_enc_file = os.path.join(base_path, config['extension'] + '_encoder.pkl')
    cat_feats_file = os.path.join(base_path, config['extension'] + '_cat_feats.pkl')
    data_types_file = os.path.join(base_path, config['extension'] + '_dtypes.pkl')
    # Training Scope: Select Categorical, Fit/Transform; Save data types, categorical features and encoder
    if scope == 'train':
        # Pickle Data Types
        log.info('\tPickling data types for prediction service....')
        pickle.dump(df.dtypes.to_dict(), open(data_types_file, 'wb'))
        # Identifying Categorical Columns
        numeric_features = df.select_dtypes(['int', 'float']).columns.tolist()
        categorical_features = df.columns.difference(numeric_features).tolist()
        # Pickling Categorical Columns
        pickle.dump(categorical_features, open(cat_feats_file, 'wb'))
    # Test Only (or) Prediction: Load encoder, data types and categorical feature; Perform encode transform
    else:
        df_dtypes = pickle.load(open(data_types_file, 'rb'))
        log.info('\tConverting data types prior to encode transformation....')
        # Is this even needed now? encoder from pd.factorize always converts to object
        for col in df.columns.tolist():
            df[col] = df[col].astype(df_dtypes[col])
        categorical_features = pickle.load(open(cat_feats_file, 'rb'))
        numeric_features = df.columns.difference(categorical_features).tolist()
        enc = pickle.load(open(out_enc_file, 'rb'))
    # Treating Empty Columns as Missing; Allows differentiation of new categorical values during test/prediction
    df[categorical_features] = df[categorical_features].fillna('Missing')
    if scope == 'train':
        encoded_df, enc = fit_transform(df, categorical_features)
        # Pickling Encoder
        pickle.dump(enc, open(out_enc_file, 'wb'))
    else:
        encoded_df = transform(df, categorical_features, enc)
    # Combine Numeric and Categorical Features
    combined_df = pd.concat([encoded_df, df[numeric_features]], axis=1, join_axes=[encoded_df.index])
    if transopts['debug']:
        out_csv_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_encoded.csv')
        pre_csv_file = os.path.join(paths['EOUTDIR'], config['extension'] + '_preencoded.csv')
        log.info('\tDumping pre-encoded CSV....')
        df.to_csv(pre_csv_file, index=False)
        log.info('\tDumping post-encoded CSV....')
        pd.concat([encoded_df, df], axis=1, join_axes=[encoded_df.index]).to_csv(out_csv_file, index=False)
    log.info('Function End')
    return combined_df
