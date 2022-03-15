FROM ubuntu:16.04

MAINTAINER Jigish Parikh <jparikh@infiniticg.com>

RUN apt-get update && apt-get install -y python-pip python3-pip python-dev python-tk python-flask cron jq wget

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Temp Fix for Rancher Dereg Script Issue
RUN cd /opt \
  && wget "https://github.com/rancher/cli/releases/download/v0.6.4/rancher-linux-amd64-v0.6.4.tar.gz" \
  && tar -xvf rancher-linux-amd64-v0.6.4.tar.gz \
  && chmod 755 -R rancher-v0.6.4 \
  && ln -s /opt/rancher-v0.6.4/rancher /usr/local/bin/rancher

COPY ./ccctc_spam/ /opt/ccctc_spam/

COPY ./devops/ /opt/devops/

ARG ML_HOME=/opt/ccctc_spam

RUN mkdir /opt/env ${ML_HOME}/project/etl ${ML_HOME}/project/logs ${ML_HOME}/project/src ${ML_HOME}/project/out ${ML_HOME}/project/out/final ${ML_HOME}/project/out/test ${ML_HOME}/project/out/train

WORKDIR /opt/env

RUN echo "Y" | pip install --upgrade pip==9.0.3

RUN echo "y" | pip install virtualenv

RUN echo "y" | pip install awscli

RUN echo "y" | pip3 install rsa

RUN virtualenv py27_spam -p /usr/bin/python2.7

RUN source py27_spam/bin/activate && pip install numpy==1.14.0 && pip install -r /opt/ccctc_spam/requirements.txt

#ENTRYPOINT /opt/devops/start_prediction_service.sh
#ENTRYPOINT /opt/devops/start_training_service.sh
