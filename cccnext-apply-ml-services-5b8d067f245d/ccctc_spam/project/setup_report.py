#!/usr/bin/env python

################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

# Standard Packages
import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

# Import Projects Constants/Parameters
from lib.utils import support_functions as sf

# Get Script Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weekly', help='Create weekly spam filter report.', action='store_true')
    parser.add_argument('--monthly', help='Create weekly spam filter report.', action='store_true')
    args = parser.parse_args()
    return args


# Load Paths
def load_paths():
    # Global Variables
    # Root Identifier - Changes by Repo
    paths = dict()
    paths['ROOTDIR'] = os.environ['ML_HOME']
    sys.path.append(paths['ROOTDIR'])

    # Global Directories
    paths['UTILDIR'] = os.path.join(paths['ROOTDIR'], 'lib/utils')
    paths['VIZDIR'] = os.path.join(paths['ROOTDIR'], 'lib/visualization')
    paths['MODELDIR'] = os.path.join(paths['ROOTDIR'], 'models')
    sys.path.append(paths['UTILDIR'])
    sys.path.append(paths['MODELDIR'])
    sys.path.append(paths['VIZDIR'])

    # Current Project Directory
    paths['PROJ'] = os.path.join(paths['ROOTDIR'], 'project')
    sys.path.append(paths['PROJ'])

    # Project Sub Directories
    paths['MOUTDIR'] = os.path.join(paths['PROJ'], 'out')
    paths['TRAINDIR'] = os.path.join(paths['MOUTDIR'], 'train')
    paths['TESTDIR'] = os.path.join(paths['MOUTDIR'], 'test')
    paths['FINALDIR'] = os.path.join(paths['MOUTDIR'], 'final')
    paths['CFGDIR'] = os.path.join(paths['PROJ'], 'cfg')
    paths['LOGDIR'] = os.path.join(paths['PROJ'], 'logs')
    paths['EOUTDIR'] = os.path.join(paths['PROJ'], 'etl')
    paths['SRCDIR'] = os.path.join(paths['PROJ'], 'src')

    # Check for DB Results file
    if 'ML_DB' in os.environ:
        paths['ML_DB'] = os.environ['ML_DB']
    else:
        paths['ML_DB'] = os.path.join(paths['TESTDIR'], 'results_db.csv')

    return paths


# Load Project Configuration Parameters
def load_parameters(paths, yf):
    yfile = os.path.join(paths['CFGDIR'], yf)
    with open(yfile, 'r') as f:
        try:
            params = yaml.load(f)
            return params
        except yaml.YAMLError as exc:
            print(exc)


# Setup Environment: Load/Update Paths, Configurations and Logger
def setup_env(args, log=None):
    # Load Project Hierarchical Paths
    paths = load_paths()

    # Load Project Parameters
    cfgfile = 'ccctc_spam_report.yaml'
    params = load_parameters(paths, cfgfile)



    return log, paths, params
