################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################


#########################################################################################################
#  Description: Low-level functions for running test on classification models and model ensembles
#
#########################################################################################################
# RandomForest is default model specified in some methods. Might not be used by high-level functions
from sklearn.ensemble import RandomForestClassifier as RandomForest

# sklearn Toolkit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, ShuffleSplit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
import pickle

from lib.utils import modeling_tools
from lib.utils import support_functions as sf

#########################################################################################################
# Setup logging
log = logging.getLogger('info')
#########################################################################################################


# Wrapper function to take input features and output from files with specific problems to solve and
# call run_models for CV and / or test
def run_models_wrapper(x, y, xtest, ytest, model=None, run_cv_flag=False, num_model_iterations=1,
                       plot_learning_curve=False, run_prob_predictions=True, return_yprob=False,
                       classification_threshold=0.5, model_class=RandomForest, **kwargs):
    log.info('Function Start')
    if run_cv_flag:
        # Run model on cross-validation dataproc
        log.info(sf.Color.BOLD + sf.Color.GREEN + "Running Cross-Validation" + sf.Color.END)
        run_model(cv_0_test_1=0, x=x, y=y, xtest=xtest, ytest=ytest, model=model,
                  num_model_iterations=num_model_iterations, run_prob_predictions=run_prob_predictions,
                  return_yprob=return_yprob, classification_threshold=classification_threshold,
                  plot_learning_curve=plot_learning_curve, model_class=model_class, **kwargs)

    # Run model on test data
    log.info(sf.Color.BOLD + sf.Color.GREEN + "Running Test" + sf.Color.END)
    results_df = run_model(cv_0_test_1=1, x=x, y=y, xtest=xtest, ytest=ytest, model=model,
                           num_model_iterations=num_model_iterations,
                           run_prob_predictions=run_prob_predictions, return_yprob=return_yprob,
                           classification_threshold=classification_threshold,
                           model_class=model_class, **kwargs)
    log.info('Function End')
    return results_df


def run_model(cv_0_test_1, x, y, xtest, ytest, model=None, num_model_iterations=1, cv_size=0.2,
              plot_learning_curve=False,
              run_prob_predictions=False, return_yprob=False, classification_threshold=0.5, model_class=RandomForest,
              **kwargs):
    # # @brief: For cross-validation, runs the model and gives accuracy and precision / recall by treating
    # #         a random sample of train data as test data
    # # @param: x - Input features (numpy array)
    # #         y - expected output (numpy array)
    # #         plot_learning_curve (only for cv) - bool
    # #         num_model_iterations - Times to run the model (to average the results)
    # #         cv_size (only for test) - % of data that should be treated as test (in decimal)
    # #         model_class - Model to run (if specified model doesn't run,
    # #                     then it'll have to be imported from sklearn)
    # #         **kwargs  - Model inputs, refer sklearn documentation for your model to see available parameters
    # #         plot_learning_curve - bool
    # # @return: None
    log.info('Function Start')
    # Create train / test split only for cv
    if cv_0_test_1:  # Run test
        y_actual = y_predicted = ytest.copy()

        x_train = np.array(x)
        x_test = np.array(xtest)
        y_train = np.array(y)
    else:  # Run cv
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cv_size, random_state=42)
        y_actual = y_predicted = y_test.copy()

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)

    # Plot learning curve only for cv
    if not cv_0_test_1 and plot_learning_curve:
        title = "Learning Curves"
        # Cross validation with 25 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

        modeling_tools.plot_learning_curve(model_class(**kwargs), title, x, y, cv=cv, n_jobs=-1)

        # plt.show()
        plot_file = model_class.__name__ + '_learning_curve.png'
        plt.savefig(plot_file)

    # Accuracy
    mean_correct_positive_prediction = 0
    mean_correct_negative_prediction = 0
    mean_incorrect_positive_prediction = 0
    mean_incorrect_negative_prediction = 0
    mean_accuracy = 0

    # Precision / Recall
    beta = 1.0  # higher beta prioritizes recall more than precision, default is 1
    mean_precision = 0
    mean_recall = 0
    mean_fbeta_score = 0

    if return_yprob:
        num_model_iterations = 1  # for probabilities returned, just run 1 iteration

    for _ in range(num_model_iterations):
        y_predicted = run_test(x_train=x_train, y_train=y_train, x_test=x_test, model=model,
                               run_prob_predictions=run_prob_predictions, return_yprob=return_yprob,
                               classification_threshold=classification_threshold,
                               model_class=model_class, **kwargs)

        # Only do accuracy / precision and recall if actual classified values are returned and not probabilities
        if not return_yprob:
            # Accuracy
            mean_accuracy += accuracy(y_actual, y_predicted)

            mean_correct_positive_prediction += correct_positive_prediction
            mean_correct_negative_prediction += correct_negative_prediction
            mean_incorrect_positive_prediction += incorrect_positive_prediction
            mean_incorrect_negative_prediction += incorrect_negative_prediction

            # Precision recall
            prec_recall = precision_recall_fscore_support(y_true=y_actual, y_pred=y_predicted, beta=beta,
                                                          average='binary')

            mean_precision += prec_recall[0]
            mean_recall += prec_recall[1]
            mean_fbeta_score += prec_recall[2]

    # Only do accuracy / precision and recall if actual classified values are returned and not probabilities
    if not return_yprob:
        # Accuracy
        mean_accuracy /= num_model_iterations
        mean_correct_positive_prediction /= num_model_iterations
        mean_correct_negative_prediction /= num_model_iterations
        mean_incorrect_positive_prediction /= num_model_iterations
        mean_incorrect_negative_prediction /= num_model_iterations

        # Precision recall
        mean_precision /= num_model_iterations
        mean_recall /= num_model_iterations
        mean_fbeta_score /= num_model_iterations

        # Accuracy
        log.debug(sf.Color.BOLD + sf.Color.DARKCYAN + "\nAccuracy {:.2f}".format(mean_accuracy * 100) + sf.Color.END)

        log.debug(sf.Color.BOLD + sf.Color.DARKCYAN + "\nCorrect positive prediction {:.2f}".format(
            mean_correct_positive_prediction) + sf.Color.END)
        log.debug(sf.Color.BOLD + sf.Color.DARKCYAN + "\nCorrect negative prediction {:.2f}".format(
            mean_correct_negative_prediction) + sf.Color.END)
        log.debug(sf.Color.BOLD + sf.Color.DARKCYAN + "\nIncorrect positive prediction {:.2f}".format(
            mean_incorrect_positive_prediction) + sf.Color.END)
        log.debug(sf.Color.BOLD + sf.Color.DARKCYAN + "\nIncorrect negative prediction {:.2f}".format(
            mean_incorrect_negative_prediction) + sf.Color.END)

        # Precision recall
        log.debug(sf.Color.BOLD + sf.Color.DARKCYAN + "\nPrecision {:.2f} Recall {:.2f} Fbeta-score {:.2f}".format(
            mean_precision * 100, mean_recall * 100, mean_fbeta_score * 100) + sf.Color.END)

    # compare probability predictions of the model
    if run_prob_predictions:
        log.debug("Prediction probabilities")

        compare_prob_predictions(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_actual, model=model,
                                 model_class=model_class, **kwargs)
    log.info('Function End')
    return pd.DataFrame({'y_actual': y_actual, 'y_predicted': y_predicted})


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    positive_prediction = np.array(y_true)  # Create np array of size y_true, values will be overwritten below

    global correct_positive_prediction
    global correct_negative_prediction
    global incorrect_positive_prediction
    global incorrect_negative_prediction

    correct_positive_prediction = 0
    correct_negative_prediction = 0
    incorrect_positive_prediction = 0
    incorrect_negative_prediction = 0

    for idx, value in np.ndenumerate(y_true):
        if y_true[idx] == y_pred[idx]:
            positive_prediction[idx] = 1.0
        else:
            positive_prediction[idx] = 0.0

        if y_pred[idx] == 1 and y_true[idx] == y_pred[idx]:
            correct_positive_prediction += 1
        elif y_pred[idx] == 0 and y_true[idx] == y_pred[idx]:
            correct_negative_prediction += 1
        else:
            if y_pred[idx]:
                incorrect_positive_prediction += 1
            else:
                incorrect_negative_prediction += 1

    log.debug("\nAccuracy method output\n")
    log.debug("correct_positive_prediction %d", correct_positive_prediction)
    log.debug("Incorrect_positive_prediction %d", incorrect_positive_prediction)
    log.debug("correct_negative_prediction %d", correct_negative_prediction)
    log.debug("Incorrect_negative_prediction %d", incorrect_negative_prediction)

    return np.mean(positive_prediction)


# Test on different dataset. Classify users into if they'll churn or no
# If run_prob_predictions is False, we rely on the model to give classified outputs based on the
# classification_threshold. If true, then model gives probabilities as outputs and we use a threshold to classify
# them into different classes. Currently probability predictions only support 2 classes
def run_test(x_train, y_train, x_test, model=None, run_prob_predictions=False, return_yprob=False,
             classification_threshold=0.5,
             model_class=RandomForest, **kwargs):
    log.info('Function Start')
    y_pred = np.zeros((len(x_test), 1), dtype=int)

    # Initialize y_prob for predicting probabilities
    y_prob = np.zeros((len(x_test), 2), dtype=float)

    time.sleep(5)  # sleep time in seconds

    if not run_prob_predictions:
        if not model:
            # TODO: Not sure what value this brings. Here for now.
            for iter_num in range(1, 2):
                # For models with n_estimators, increase it with iteration
                # (useful when warm_start=True for these models)
                # try:
                #     model.set_params(n_estimators=100 * iter_num)
                # except ValueError:
                #     log.debug("Model does not have n_estimators")
                model = model_class(**kwargs)
                log.debug(model)
                model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    else:  # Predict probabilities
        if not model:
            model = model_class(**kwargs)
            log.debug(model)
            model.fit(x_train, y_train)
        # y_prob[idx, class]. Since classes are 2 here, will contain info on prob of both classes
        y_prob = model.predict_proba(x_test)
        log.debug(y_prob)

    if hasattr(model, "feature_importances_"):
        log.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        log.debug(model.feature_importances_)

    # Print list of predicted classes in order
    if hasattr(model, "classes_"):
        log.debug(sf.Color.BOLD + sf.Color.BLUE + "Predict probability classes" + sf.Color.END)
        log.debug(model.classes_)

    # log.info(model.estimators_)

    if run_prob_predictions:
        for idx, _ in np.ndindex(y_prob.shape):
            # Column 1 has the predicted y_prob for class "1"
            if y_prob[idx, 1] < classification_threshold:
                y_pred[idx] = 0
            else:
                y_pred[idx] = 1

    y_pred = np.array(y_pred)

    if not run_prob_predictions and return_yprob:
        raise ValueError("Invalid combination - cannot return yprob when run_prob_predictions is False!")

    log.info('Function End')
    if return_yprob:
        # Column 1 has the predicted y_prob for class "1"
        return y_prob[:, 1]
    else:
        return y_pred


# Test to compare probabilities of the predictions vs. just prediction accuracy
def compare_prob_predictions(x_train, y_train, x_test, y_test, model_class, model=None, **kwargs):
    # import warnings
    # warnings.filterwarnings('ignore')  # TODO - check if we can remove this
    log.info('Function Start')
    # Use 10 estimators (inside run_cv and run_test so predictions are all multiples of 0.1
    pred_prob = run_test(x_train=x_train, y_train=y_train, x_test=x_test, model=model,
                         run_prob_predictions=True, return_yprob=True,
                         classification_threshold=0.5, model_class=model_class, **kwargs)

    is_output_1 = (y_test == 1)

    log.debug(pred_prob)

    # Number of times a predicted probability is assigned to an observation
    counts = pd.value_counts(pred_prob).sort_index()

    # calculate true probabilities
    true_prob = {}

    log.debug(counts)

    for prob in counts.index:
        # Pep8 shows a warning that's not valid
        true_prob[prob] = np.mean(is_output_1[pred_prob == prob])
        true_prob = pd.Series(true_prob)

    counts = pd.concat([counts, true_prob], axis=1).reset_index()

    counts.columns = ['pred_prob', 'count', 'true_prob']
    log.debug(counts)
    # print ("Num_wrong_predictions")
    # print (1.0 - counts.icol(0)) * counts.icol(1) * counts.icol(2)
    log.info('Function End')

##################################################################################################################
