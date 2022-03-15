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
from logging.config import fileConfig
from datetime import datetime

# Import Projects Constants/Parameters
from lib.utils import support_functions as sf


# Setup logging
def setup_logging(args, paths, params, prefix):
    logbase = prefix + '_'
    logging_config_file = os.path.join(paths['UTILDIR'], 'logging.conf')
    logging.captureWarnings(True)
    logging.config.fileConfig(logging_config_file, disable_existing_loggers=False)
    log = logging.getLogger('info')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(module)s: %(funcName)s(): %(asctime)s] %(levelname)s: %(message)s')
    if not args.disable_log:
        # run_tstmp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_tstmp = sf.get_timestamp(tzinfo='US/Pacific')
        logname = os.path.join(paths['LOGDIR'], logbase + run_tstmp + '.log')
        params['LOGNAME'] = logname
        params['RUN_TSTMP'] = run_tstmp
        fh = logging.FileHandler(logname)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log, params


# Get Script Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', help='Extract source data for analytics', action='store_true')
    parser.add_argument('--transform', help='Transform source data for analytics', action='store_true')
    parser.add_argument('--test', help='Run model on test data', action='store_true')
    parser.add_argument('--train', help='Train model', action='store_true')
    parser.add_argument('--train_src', help='Source File (including Path) for train', type=argparse.FileType('rt'))
    parser.add_argument('--test_src', help='Source File (including Path) for test', type=argparse.FileType('rt'))
    parser.add_argument('--dest', help='Destination Directory for training outputs', action='store')
    parser.add_argument('--cfg', help='Override default config', action='store')
    parser.add_argument('--tag', help='Tag experiment/run name. Allows better tracking of results in database', action='store')
    parser.add_argument('--predict', help='Generate prediction for input JSON', action='store')
    parser.add_argument('--disable_log', help='Disable log file from getting dumped', action='store_true')
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
    if args.cfg:
        cfgfile = args.cfg
    else:
        cfgfile = 'ccctc_spam.yaml'
    params = load_parameters(paths, cfgfile)

    # Ensure --extract --transform and --train are used for production training in production freeze
    if params['production']['freeze'] and not params['production']['test_mode'] and ('ccctc_spam.py' in sys.argv[0]):
        assert (args.extract and args.transform and args.train), \
            'Production Mode Enabled: Need to add --extract --transform --train!'

    # Ensure --test and params['test']['split_size'] == 0 are not set at the same time
    if params['test']['split_size'] == 0 and ('ccctc_spam.py' in sys.argv[0]) and not args.test_src:
        assert (not args.test), 'Cannot enable --test with split_size of zero'

    # Ensure params['train']['split_size'] == 0 doesn't happen with training
    if args.train and 'ccctc_spam.py' in sys.argv[0]:
        assert (params['train']['split_size'] != 0), \
            'Cannot have training with split_size of zero. Use train | retrain to train model with full data-set'

    # Isolating Config Parameters
    config = params['config']

    # Setup Logging
    if not log:
        log, params = setup_logging(args, paths, params, config['extension'])
    log.info('\n\nRun Configuration Parameters\n{}\n'.format(sf.tabulate_dict(config)))

    # Log commandline
    log.info('Run Command: \n\n{}\n'.format(' '.join(sys.argv)))

    # Update Train Source Path
    if args.train_src:
        log.info('Loading commandline path to input data: {}'.format(args.train_src))
        paths['SRCFILE'] = args.train_src
    elif 'SRCFILE' in os.environ:
        log.info('Loading environment variable source to input data: {}'.format(os.environ['SRCFILE']))
        paths['SRCFILE'] = os.environ['SRCFILE']
    else:
        paths['SRCFILE'] = os.path.join(paths['SRCDIR'], config['extension'] + '.csv')

    # Update Test Source Path
    if args.test_src:
        log.info('Loading commandline path to input data: {}'.format(args.test_src))
        paths['TEST_SRC'] = args.test_src

    # Update Destination Path/Folder
    if args.dest:
        log.info('Loading commandline path to output folder: {}'.format(args.dest))
        paths['FINALDIR'] = args.dest
    elif 'DESTDIR' in os.environ:
        log.info('Loading environment variable for destination folder: {}'.format(os.environ['DESTDIR']))
        paths['FINALDIR'] = os.environ['DESTDIR']

    return log, paths, params
