################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

#########################################################################################################
#  Description: Low-level functions for running cross-validation on classification models and model ensembles
#
#########################################################################################################
# RandomForest is default model specified in some methods. Might not be used by high-level functions
from sklearn.ensemble import RandomForestClassifier as RandomForest

# sklearn Toolkit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging

from lib.utils import modeling_tools
from lib.utils import support_functions as sf

#########################################################################################################
# Setup logging
log = logging.getLogger('info')
#########################################################################################################


# Wrapper function to take input features and output from files with specific problems to solve and
# call run_models for CV and / or test
def run_models_wrapper(x, y, run_cv_flag=False, num_model_iterations=1, plot_learning_curve=False,
                       run_prob_predictions=True, return_yprob=False, test_size=0.2,
                       classification_threshold=0.5, model_class=RandomForest, **kwargs):
    log.info('Function Start')
    if run_cv_flag:
        # Run model on cross-validation dataproc
        log.info(sf.Color.BOLD + sf.Color.GREEN + "Running kfold Cross-Validation" + sf.Color.END)
        model, _, _ = run_model(cv_0_test_1=0, x=x, y=y, num_model_iterations=num_model_iterations,
                                test_size=test_size, run_prob_predictions=run_prob_predictions,
                                return_yprob=return_yprob, classification_threshold=classification_threshold,
                                plot_learning_curve=plot_learning_curve, model_class=model_class, **kwargs)
    # Run model on test data
    log.info(sf.Color.BOLD + sf.Color.GREEN + "Running Cross-Validation" + sf.Color.END)
    model, y_actual, results_df = run_model(cv_0_test_1=1, x=x, y=y,
                                            num_model_iterations=num_model_iterations,
                                            test_size=test_size, run_prob_predictions=run_prob_predictions,
                                            return_yprob=return_yprob,
                                            classification_threshold=classification_threshold,
                                            model_class=model_class, **kwargs)
    log.info('Function End')
    return model, y_actual, results_df


def run_model(cv_0_test_1, x, y, num_model_iterations=1, test_size=0.2, plot_learning_curve=False,
              run_prob_predictions=False, return_yprob=False, classification_threshold=0.5, model_class=RandomForest,
              **kwargs):
    # # @brief: For cross-validation, Runs the model and gives accuracy and precision / recall
    # #         For test, runs the model and gives accuracy and precision / recall by treating
    # #         a random sample of input dataproc as test dataproc
    # # @param: x - Input features (numpy array)
    # #         y - expected output (numpy array)
    # #         plot_learning_curve (only for cv) - bool
    # #         num_model_iterations - Times to run the model (to average the results)
    # #         test_size (only for test) - % of dataproc that should be treated as test (in decimal)
    # #         model_class - Model to run (if specified model doesn't run,
    # #                     then it'll have to be imported from sklearn)
    # #         **kwargs  - Model inputs, refer sklearn documentation for your model to see available parameters
    # #         plot_learning_curve - bool
    # # @return: None
    log.info('Function Start')
    # Create train / test split only for test
    if cv_0_test_1:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        y_actual = y_predicted = y_test.copy()

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
    else:
        x_train, x_test, y_train, y_test = 0, 0, 0, 0

        y_actual = y_predicted = y.copy()

    # Plot learning curve only for cv
    if not cv_0_test_1 and plot_learning_curve:
        title = "Learning Curves"
        # Cross validation with 25 iterations to get smoother mean test and train
        # score curves, each time with 20% dataproc randomly selected as a validation set.
        # cv = ShuffleSplit(x.shape[0], n_splits=25, test_size=0.2, random_state=0)
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
        if cv_0_test_1:  # test
            model, y_predicted = run_test(x_train=x_train, y_train=y_train, x_test=x_test,
                                          run_prob_predictions=run_prob_predictions, return_yprob=return_yprob,
                                          classification_threshold=classification_threshold,
                                          model_class=model_class, **kwargs)
        else:  # cv
            model, y_predicted = run_cv(x=x, y=y, run_prob_predictions=run_prob_predictions, return_yprob=return_yprob,
                                        classification_threshold=classification_threshold, model_class=model_class,
                                        **kwargs)

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
    # if run_prob_predictions:
    #     if not cv_0_test_1:
    #         log.debug("Prediction probabilities for CV")
    #         compare_prob_predictions(cv_0_test_1=cv_0_test_1, x_train=x, y_train=y, x_test=0, y_test=y_actual,
    #                                  model_class=model_class, **kwargs)
    #     else:
    #         log.debug("Prediction probabilities for Test")
    #         compare_prob_predictions(cv_0_test_1=cv_0_test_1, x_train=x_train, y_train=y_train, x_test=x_test,
    #                                  y_test=y_actual, model_class=model_class, **kwargs)
    log.info('Function End')
    return model, y_actual, pd.DataFrame({'y_actual': y_actual, 'y_predicted': y_predicted})


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


# Run k-fold cross-validation. Classify users into if they'll churn or no
# If run_prob_predictions is False, we rely on the model to give classified outputs.
# If true, then model gives probabilities as outputs and we use a threshold to classify them into
# different classes. Currently probability predictions only support 2 classes
def run_cv(x, y, run_prob_predictions=False, return_yprob=False, classification_threshold=0.5,
           model_class=RandomForest, **kwargs):
    log.info('Function Start')
    # Construct a kfolds object
    kf = KFold(n_splits=5, shuffle=False)

    y_pred = y.copy()
    y_prob = np.zeros((len(y), 2), dtype=float)

    # log.debug(kf)
    # Initialize to avoid pep8 warning, thought model will always be initialized below
    model = 0

    # Iterate through folds
    for train_index, test_index in kf.split(x):
        # Initialize a classifier with key word arguments
        model = model_class(**kwargs)
        log.debug('TRAIN: {}; TEST: {}'.format(train_index, test_index))
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        if not train_index[0]:
            log.debug(model)

        # if run_prob_predictions:
        #     # For probability prediction, just run 10 estimators
        #     model.set_params(n_estimators=10)

        model.fit(x_train, y_train)

        if not run_prob_predictions:
            y_pred[test_index] = model.predict(x_test)
        else:  # Predict probabilities
            # y_prob[idx, class]. Since classes are 2 here, will contain info on prob of both classes
            y_prob[test_index] = model.predict_proba(x_test)

            # accuracy(y[test_index], y_pred[test_index])

    if hasattr(model, "feature_importances_"):
        log.info(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        log.info(model.feature_importances_)

    # log.info(model.estimators_)

    if run_prob_predictions:
        for idx, _ in np.ndindex(y_prob.shape):
            if y_prob[idx, 1] < classification_threshold:
                y_pred[idx] = 0
            else:
                y_pred[idx] = 1
                # print y_prob

    if not run_prob_predictions and return_yprob:
        raise ValueError("Invalid combination - cannot return yprob when run_prob_predictions is False!")

    log.info('Function End')
    if return_yprob:
        # Column 1 has the predicted y_prob for class "1"
        return model, y_prob[:, 1]
    else:
        return model, y_pred


# Test on different dataset. Classify users into if they'll churn or no
# If run_prob_predictions is False, we rely on the model to give classified outputs based on the
# classification_threshold. If true, then model gives probabilities as outputs and we use a threshold to classify
# them into different classes. Currently probability predictions only support 2 classes
def run_test(x_train, y_train, x_test, run_prob_predictions=False, return_yprob=False, classification_threshold=0.5,
             model_class=RandomForest, **kwargs):
    log.info('Function Start')
    y_pred = np.zeros((len(x_test), 1), dtype=int)

    # Initialize y_prob for predicting probabilities
    y_prob = np.zeros((len(x_test), 2), dtype=float)

    # Initialize a classifier with key word arguments
    model = model_class(**kwargs)

    log.info(model)

    time.sleep(5)  # sleep time in seconds

    if not run_prob_predictions:
        for iter_num in range(1, 2):
            # For models with n_estimators, increase it with iteration (useful when warm_start=True for these models)
            # try:
            #     model.set_params(n_estimators=100 * iter_num)
            # except ValueError:
            #     log.debug("Model does not have n_estimators")

            # log.debug(model)
            model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    else:  # Predict probabilities
        # log.debug(model)
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
        # Print importance of the input features and probability for each prediction
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
        return model, y_prob[:, 1]
    else:
        return model, y_pred


# Run k-fold cross-validation. Classify users into if they'll churn or no
def run_cv_splits(x, y, num_of_splits, model_class, **kwargs):
    log.info('Function Start')
    if num_of_splits < 2:
        raise ValueError("Invalid number of splits, needs to be atleast 2")

    # Construct a list with different x, y for each split
    # Initialize empty lists
    x_split = []
    y_split = []

    num_rows = x.shape[0] + 1  # TODO - check if equ to np.shape(x)[0] which gives a warning

    for split in range(num_of_splits):
        start_row = split * num_rows / num_of_splits
        end_row = ((split + 1) * num_rows / num_of_splits)

        x_split.append(x[start_row:end_row, :])
        y_split.append(y[start_row:end_row])

    # Make the last split the test dataproc
    x_test = x_split[num_of_splits - 1]
    y_test = y_split[num_of_splits - 1]

    # Initialize to avoid pep8 warning, thought model and y_pred will always be initialized below
    y_pred = model = 0

    # Iterate through first n-1 splits
    for split in range(num_of_splits - 1):
        x_train = x_split[split]
        y_train = y_split[split]

        # Initialize a classifier with key word arguments
        model = model_class(**kwargs)

        if not split:  # Print just once
            log.debug(model)

        m = model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy(y_test, y_pred)

    if hasattr(model, "feature_importances_"):
        log.debug(sf.Color.BOLD + sf.Color.BLUE + "\nFeature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        log.debug(model.feature_importances_)
    log.info('Function End')
    return m, [y_test, y_pred]


# Test to compare probabilities of the predictions vs. just prediction accuracy
def compare_prob_predictions(cv_0_test_1, x_train, y_train, x_test, y_test, model_class, **kwargs):
    log.info('Function Start')
    import warnings
    warnings.filterwarnings('ignore')  # TODO - check if we can remove this

    # Use 10 estimators (inside run_cv and run_test so predictions are all multiples of 0.1
    if not cv_0_test_1:  # Run CV
        # TODO: [Ananth]: Changed return_yprob to True; Was forced to False.
        # Not using True returned boolean responses which failed the np.mean computations below
        m, pred_prob = run_cv(x=x_train, y=y_train, run_prob_predictions=True, return_yprob=True,
                              classification_threshold=0.5, model_class=model_class, **kwargs)
    else:  # Run test
        m, pred_prob = run_test(x_train=x_train, y_train=y_train, x_test=x_test, run_prob_predictions=True,
                                return_yprob=True, classification_threshold=0.5, model_class=model_class, **kwargs)

    is_output_1 = (y_test == 1)

    # log.debug(pred_prob)

    # Number of times a predicted probability is assigned to an observation
    counts = pd.value_counts(pred_prob).sort_index()

    # calculate true probabilities
    true_prob = {}

    # log.debug(counts)

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
