from __future__ import absolute_import, division, print_function, unicode_literals

from flask import Flask, jsonify, request, Response
import dotenv
import os
import pandas as pd
import logging
import urllib2
import json
from jwcrypto import jwk, jwt

from ccctc_spam import predict_fraud
from project.setup import setup_env

##################################################################################################################
# Setup Logging
logging.config.fileConfig('./lib/utils/logging.conf', disable_existing_loggers=False)
log = logging.getLogger('info')
##################################################################################################################


# Dummy class created to store arguments
class Arguments:
    pass


# Function to override using argparse
# argparse cli conflicts with gunicorn
def load_args():
    args = Arguments()
    args.extract = False
    args.transform = False
    args.test = False
    args.train = False
    args.train_src = False
    args.test_src = False
    args.dest = False
    args.cfg = False
    args.predict = False
    if os.environ.get('ML_ENV') != 'prod':
        args.disable_log = False
    else:
        args.disable_log = True
    return args


app = Flask(__name__)
# dotenv.load()
dotenv.load_dotenv('./.env')

# Set up environment
args = load_args()
log, paths, params = setup_env(args)
transopts = params['load']['transform']
extropts = params['load']['extract']
apiopts = params['api']
stored_token = ''

# Extract Keyset
if apiopts['enable_auth']:
    mitre_url = os.environ.get('ML_MITRE_URL')
    log.info('Accessing keys for authentication. URL: {}'.format(mitre_url))
    key_json = urllib2.urlopen(mitre_url).read()
    log.debug('Key: {}'.format(key_json))
    key_set = jwk.JWKSet.from_json(key_json)
    log.debug('KeySet: {}'.format(key_set))


@app.route('/')
def index():
    return 'API is working!'


@app.route('/health')
def health():
    return 'UP'


@app.route('/ml', methods=['POST'])
def ml():
    global stored_token
    y_col = transopts['y_column']
    json_input = request.json
    if apiopts['enable_auth']:
        token = request.headers['Authorization'].split()[1]
        if token != stored_token:
            stored_token = token
            log.info('Authenticating token.....')
            log.debug('Token Extracted: {}'.format(token))
            try:
                token_val = jwt.JWT(jwt=token, key=key_set)
                log.debug('Token Value: {}'.format(token_val))
            except ValueError:
                return Response('Token failed validation', status=400, mimetype='text/HTML')
    if not isinstance(json_input, list):
        json_object = [json_input]
    else:
        json_object = json_input

    pred_df = pd.DataFrame(columns=['app_id', 'pred_fraud_prob', 'pred_fraud'])
    # Logging call in lower environments
    if os.environ.get('ML_ENV') != 'prod':
        log.info('Application Data: \n{}'.format(json.dumps(json_object)))
    # Override Prediction Service Output
    if os.environ.get('ML_PREDICTION_BYPASS_FLAG') == 'True':
        log.info('ML_PREDICTION_BYPASS_FLAG set to True. Prediction Service Bypassed! Default Response: Not Fraud')
        # Convert JSON to dataframe
        df = pd.DataFrame(json_object)
        pred_df['app_id'] = df['app_id']
        pred_df['pred_fraud'] = 0
        pred_df['pred_fraud_prob'] = 0.0
    else:
        pred_df[['pred_fraud_prob', 'pred_fraud', 'app_id']] = predict_fraud(args, paths, params, json_obj=json_object)
    if y_col in json_object:
        pred_df[y_col] = int(json_object[y_col])
    log.info('Prediction: {}'.format(pred_df.to_json(orient='index')))
    resp = Response(pred_df.to_json(orient='index'), status=200, mimetype='application/json')
    return resp


log.info(os.environ.get('PORT'))


if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0', port=dotenv.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT'))
