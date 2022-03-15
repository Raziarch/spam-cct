# This is the root directory for the project

1. In order to configure the project to work, please set environment variable $spam
   	Add the following line to your ~/.basrhc
		export ML_SPAM='<path/to/ccctc_spam>'

2. $spam/project/src is the directory that contains the input CSV
	The input file should be named ccctc_spam.csv
	Alternatively you can add --train_src <path> to point to a specific file


--------------------
For Initial Testing
--------------------

1. To test if the input data is loading & transforming:
	cd $ML_SPAM
	python ccctc_spam.py --extract --transform

2. To train the model:
	cd $ML_SPAM
	python ccctc_spam.py --train

3. To test the model through the API
	cd $ML_SPAM
	python cctcc_spam_api.py
	Make api calls with the payload through postman (or) other REST clients

--------------------
For Production
-------------------

1. Train the model
	cd $ML_SPAM
	python ccctc_spam.py --extract --transform --train --cfg ccctc_spam_gold.yaml
	<Optionally you can add --train_src to point to a different file/path>

2. Prediction
	cd $ML_SPAM
	# To kick-off the API program
	python ccctc_spam_api.py --disable_log --cfg ccctc_spam_gold.yaml
 
