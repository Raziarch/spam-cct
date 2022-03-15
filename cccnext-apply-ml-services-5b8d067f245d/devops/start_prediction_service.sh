#!/bin/bash

#home directory
cd /opt/ccctc_spam

## docker crontab does not have access to ENV variable so creating this file and sourcing it in crontab
printenv | sed 's/^\(.*\)$/export \1/g' | grep -E "^export ML" > /opt/devops/project_env.sh

## Running sync process first time before launching service
/opt/devops/ml_sync_model_files_from_s3.sh

## adding crontab to sync model files from s3
cp /opt/devops/ml_crontab /etc/cron.d/ml_sync_model_files_from_s3
chmod 755 /etc/cron.d/ml_sync_model_files_from_s3
touch /var/log/cron.log

## starting crontab service in container
service cron start

## activating virtual environment for prediction service
source /opt/env/py27_spam/bin/activate

## running the service in virtual environment
python /opt/ccctc_spam/ccctc_spam_api.py
