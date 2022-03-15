#!/bin/bash

# Rancher info
rancherURL="http://rancher-server.ccctechcenter.org:8080/v2-beta/schemas"

# Get Rancher Login Credentials using AWS SSM Property Store
function getRancherLoginCmdStr(){

        rancher_accesskey=$(aws ssm get-parameters --names rancher-key-apply-${ML_ENV} --with-decryption --region us-west-2 | jq -r '.Parameters[].Value')
        rancher_secretkey=$(aws ssm get-parameters --names rancher-pass-apply-${ML_ENV} --with-decryption --region us-west-2 | jq -r '.Parameters[].Value')
        cmdString="rancher --env apply-${ML_ENV} --access-key $rancher_accesskey --secret-key $rancher_secretkey --url $rancherURL"
}

# Removing disconnected training service host if found in Rancher
function removeDisconectedHost(){

    hostId=$($cmdString hosts ls | grep apply-ml-training-ec2-${ML_ENV} | grep -i disconnected | awk '{print $1}')

    if [ "$hostId" != "" ]; then
     echo "Removing disconnected host from Rancher" + $hostId
     $cmdString deactivate $hostId
     $cmdString rm $hostId
    else
     echo "no disconnected host found for ML Training service"
    fi

}


# Removing inactive training service host if found in Rancher
function removeInactiveHost(){

    hostId=$($cmdString hosts --all | grep apply-ml-training-ec2-${ML_ENV} | grep -i inactive | awk '{print $1}')

    if [ "$hostId" != "" ]; then
     echo "Removing inactive host from Rancher" + $hostId
     #$cmdString deactivate $hostId
     $cmdString rm $hostId
    else
     echo "no inactive host found for ML Training service"
    fi

}


####### Main program starts here

getRancherLoginCmdStr
removeDisconectedHost
removeInactiveHost
