#!/bin/bash

ML_SERVICE="apply-ml-training"
AWS_DEFAULT_REGION="us-west-2"

function validateInputFile() {

    ## input file validation
    echo "Input File Validation Started"

    filename=$(aws s3 ls ${ML_BUCKET}/data/ | sort | grep '\.pgp$' | tail -n 1 | awk '{print $4}' )
    if [[ $filename == *.pgp ]]; then
        echo "Received Input File $filename"
        return 0
    else
        echo "Expecting *.pgp File, Recieved $filename"
        notify "ERROR" "Expecting *.pgp File, Recieved $filename"
        return 1
    fi
}

function decryptFile(){

    ## remove any older file if any (edge case scenario)
    rm -rf /opt/ccctc_spam/project/src/ccctc_spam.csv
    ## decrypt file
    echo "decrypting ${filename}"
    passwd=$(aws ssm get-parameters --names apply-ml-paraphrase-${ML_ENV} --with-decryption --region us-west-2 | jq -r '.Parameters[].Value')
    aws s3 cp ${ML_BUCKET}/config/apply-ml-private-key-${ML_ENV}.asc .
    gpg --import apply-ml-private-key-${ML_ENV}.asc
    aws s3 cp ${ML_BUCKET}/data/${filename} ${ML_BUCKET}/archive/${filename}
    aws s3 cp ${ML_BUCKET}/data/${filename} .
    echo $passwd | gpg --no-tty --passphrase-fd 0 --output /opt/ccctc_spam/project/src/ccctc_spam.csv ${filename}
#    if [[ "$?" != "0" ]]; then
#        notify "ERROR" "Decrypt Operation Failed - File Name : ${filename}"
#        return 1
#    fi

    return 0

}
## Added on 01/14/2021 to get configuration files for training service
function getConfigFiles() {

    ## get config files for training service
    echo "Getting Configuraiton File for Training Service"

    mkdir -p /opt/ccctc_spam/project/out/final
    aws s3 cp ${ML_BUCKET}/model/ /opt/ccctc_spam/project/out/final/ --recursive --exclude "*.pkl"
}

function startTraining(){

    echo "starting ML traiing service"

    ## run from home directory
    cd ${ML_HOME}

    ## activating virtual environment for training service
    source /opt/env/py27_spam/bin/activate

    ## running training service in virtual environment
    python /opt/ccctc_spam/ccctc_spam.py --extract --transform --train

    if [[ "$?" != "0" ]]; then
        notify "ERROR" "Encounter Error in Training Service"
        return 1
    fi

    return 0

}

function validateModelFiles(){

    ## pkl file validation
    echo "Training completed successfully. Validating model files."
    count=$(ls -ltr /opt/ccctc_spam/project/out/final/*.pkl | wc -l)
    if [ $count -eq 5 ]; then
      fileList=$(aws s3 ls ${ML_BUCKET}/model/ | awk '{print $4}')
      arr=( $fileList )
      today=`date +%Y%m%d.%H%M%S`
      for i in "${arr[@]}"
      do
         #archive old model files to s3 bucket
         aws s3 cp ${ML_BUCKET}/model/${i} ${ML_BUCKET}/archive/${i}_${today}
      done

      #move new model files to s3 bucket
      aws s3 cp /opt/ccctc_spam/project/out/final/ ${ML_BUCKET}/model/ --recursive --exclude "*" --include "*.pkl"
      return 0
    else
        notify "ERROR" "Expecting 5 pkl files, Exists $count"
        return 1
    fi
}

function createReports(){

    ## input file validation
    echo "Input File Validation Started"

    filename=$(aws s3 ls ${ML_BUCKET}/data/ | sort | grep '\.csv$' | tail -n 1 | awk '{print $4}')
    if [[ $filename == *.csv ]]; then
        echo "Received Input data for Reports - $filename"

        ## Remove any previous data file on container
        rm -rf ${ML_HOME}/project/src/ccctc_spam_report.csv 

        ## copy report file from s3 to src directory in container
        aws s3 cp ${ML_BUCKET}/data/${filename} ${ML_HOME}/project/src/ccctc_spam_report.csv

        echo "Creating Monthly/Weekly reports.."

        ## run from home directory
        cd ${ML_HOME}

        ## activating virtual environment for training service
        source /opt/env/py27_spam/bin/activate

        ## running training service in virtual environment
        python ccctc_spam_report.py

        if [[ "$?" != "0" ]]; then
            notify "ERROR" "Encounter Error while creating reports"
            return 1
        fi

        return 0    
    
    else
        echo "No data file found for Report"
        return 0
    fi
}

function deleteDataFile(){

    ## delete decrypted file from instance and encrypted file from bucket
    aws s3 rm ${ML_BUCKET}/data --recursive
    rm -rf ${ML_HOME}/project/src/ccctc_spam.csv
    rm -rf ${ML_HOME}/project/src/ccctc_spam_report.csv

}

function archiveFiles(){


    ## log files archieve
    if [ `ls /opt/ccctc_spam/project/logs/*.log 2> /dev/null | wc -l` != 0 ]; then
        aws s3 mv /opt/ccctc_spam/project/logs/ ${ML_BUCKET}/logs/ --recursive --exclude "*" --include "*.log"
        if [[ "$?" != "0" ]]; then
            notify "ERROR" "Error archiving logs to S3 bucket"
            return 1
        fi
    fi

    ## weekly report file archieve
    if [ `ls /opt/ccctc_spam/project/logs/weekly*.html 2> /dev/null | wc -l` != 0 ]; then
        aws s3 mv /opt/ccctc_spam/project/logs/ ${ML_BUCKET}/report/weekly --recursive --exclude "*" --include "weekly*.html"
        if [[ "$?" != "0" ]]; then
            notify "ERROR" "Error archiving weekly report to S3 bucket"
            return 1
        fi
    fi


    ## monthly report file archieve
    if [ `ls /opt/ccctc_spam/project/logs/monthly*.html 2> /dev/null | wc -l` != 0 ]; then
        aws s3 mv /opt/ccctc_spam/project/logs/ ${ML_BUCKET}/report/monthly --recursive --exclude "*" --include "monthly*.html"
        if [[ "$?" != "0" ]]; then
            notify "ERROR" "Error archiving monthly report to S3 bucket"
            return 1
        fi
    fi

    # send termination trigger to terminate training host - training is successfully completed.
    touch /opt/ccctc_spam/project/logs/dummy-`date +%F`.html
    aws s3 mv /opt/ccctc_spam/project/logs/ ${ML_BUCKET}/report/ --recursive --exclude "*" --include "dummy*.html"
    if [[ "$?" != "0" ]]; then
            notify "ERROR" "Error sending terminataion trigger to workflow"
            return 1
    fi  

    return 0

}

function notify(){

json_message=$(cat <<EOF
    {
      "env": "$ML_ENV",
      "service": "$ML_SERVICE",
      "level": "$1",
      "message": "$2"
    }
EOF
)

    messageID=$(aws sns publish --topic-arn "${ML_TOPIC_ARN}" \
                    --region "us-west-2" \
                    --message "${json_message}")
    return 0
}

##############################################
## main

validateInputFile
if [[ "$?" != "0" ]]; then
    exit 1
fi

decryptFile
if [[ "$?" != "0" ]]; then
    exit 1
fi

getConfigFiles
if [[ "$?" != "0" ]]; then
    exit 1
fi

startTraining
if [[ "$?" != "0" ]]; then
    exit 1
fi

validateModelFiles
if [[ "$?" != "0" ]]; then
    exit 1
fi

createReports
if [[ "$?" != "0" ]]; then
    exit 1
fi

deleteDataFile
if [[ "$?" != "0" ]]; then
    exit 1
fi

archiveFiles
if [[ "$?" != "0" ]]; then
    exit 1
fi
