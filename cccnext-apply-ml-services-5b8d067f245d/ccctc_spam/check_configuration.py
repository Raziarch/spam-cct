import argparse
import timeit
import yaml


# Get Script Arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='configuration file', action='store')
    args = parser.parse_args()
    return args


# Load Project Configuration Parameters
def load_parameters(yf):
    with open(yf, 'r') as f:
        try:
            params = yaml.load(f)
            return params
        except yaml.YAMLError as exc:
            print(exc)


# Check environment
def check_env(cfg):

    params = load_parameters(cfg)

    # Check that anonymized email is set to False
    assert (params['load']['transform']['anonymized_email'] is False), \
        'load|transform|anonymized_email is set to True. Email is not anonymized in production!'

    # Check all debug switches are turned off (False)
    assert (params['load']['transform']['debug'] is False), \
        'load|transform|debug is set to True. Debug mode should be disabled in production!'
    assert (params['train']['debug'] is False), \
        'train|debug is set to True. Debug mode should be disabled in production!'
    assert (params['test']['debug'] is False), \
        'test|debug is set to True. Debug mode should be disabled in production!'
    assert (params['predict']['debug'] is False), \
        'predict|debug is set to True. Debug mode should be disabled in production!'
    assert (params['production']['debug'] is False), \
        'production|debug is set to True. Debug mode should be disabled in production!'

    # Check that production|freeze is set to True
    assert (params['production']['freeze'] is True), \
        'production|freeze is set to False. Debug mode should be disabled for production!'

    # Check to ensure static run method is selected. This could change in future
    assert (params['model']['miscopts']['run_method'] == 'static'), \
        'model|miscopts|run_method is not set to static. Please change!'

    # Check that train|retrain is set to True. This could change in future
    assert (params['train']['retrain'] is True), \
        'train|retrain is not set to True. Please change!'

    # Check that test|split_size is set to zero
    assert (params['test']['split_size'] == 0), \
        'Test phase is not enabled in production. Please change test|split_size to 0 (zero)'

    # Check that api|authentication is set to True
    assert (params['api']['enable_auth'] is True), \
        'api|authentication needs to be set to True for production!'

    # Check that plots generation is set to False
    assert (params['evaluate']['train']['learning_curve'] is False), \
        'evaluate|train|learning_curve should be set to False in production!'
    assert (params['evaluate']['train']['validation_curve'] is False), \
        'evaluate|train|validation_curve should be set to False in production!'

    # Check that noise ratio is not enabled
    assert (params['evaluate']['noise_ratio'] is None), \
        'Do not set evaluate|noise_ratio is production!'

    # Do not enable load_best_params. This could change in future
    assert (params['model']['miscopts']['load_best_params'] is False), \
        'Do not enable model|miscopts|load_best_params in production!'

    # Ensure production|test_mode is set to False
    assert (params['production']['test_mode'] is False), \
        'production|test_mode should be set to False in production!'

    return "Passed"

# Main
if __name__ == "__main__":
    start = timeit.default_timer()
    # Load Arguments
    args = get_args()
    # Setup Environment
    result = check_env(args.cfg)
    print result
    stop = timeit.default_timer()
    print('run_metrics: total_duration - {}'.format(stop - start))