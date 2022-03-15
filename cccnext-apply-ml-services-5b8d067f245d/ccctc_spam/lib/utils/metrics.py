################################################################
# Copyright (C) 2017 SeyVu, Inc; support@seyvu.com
#
# The file contents can not be copied and/or distributed
# without the express permission of SeyVu, Inc
################################################################

#########################################################################################################
#  Description: Collection of loss / error metrics used by the different projects
#
#########################################################################################################
import math

#########################################################################################################
__author__ = 'DataCentric1'
__pass__ = 1
__fail__ = 0


#########################################################################################################


# LogLoss metric. Inputs are actual classification and predicted probability for each classification
# Smaller log loss is better. The use of the logarithm provides extreme punishments for being both confident
# and wrong.
def logloss(y_actual, y_predicted_prob):
    if len(y_actual) != len(y_predicted_prob):
        raise ValueError("y_actual and y_predicted_prob lengths don't match")

    number_of_samples = len(y_actual)

    print number_of_samples

    logloss_score = 0

    for i in range(number_of_samples):
        # In the worst possible case for logloss, a prediction that something is true when it is actually false will
        # add infinite to your error score. In order to prevent this, predictions are bounded away from the extremes
        # by a small value.if y_predicted_prob[i] == 0:
        if y_predicted_prob[i] <= 0:
            y_predicted_prob[i] = 1e-4

        if y_predicted_prob[i] >= 1:
            y_predicted_prob[i] = 0.9999

        logloss_score += y_actual[i] * math.log(y_predicted_prob[i]) + (1 - y_actual[i]) * math.log(
            1 - y_predicted_prob[i])

    logloss_score = -logloss_score / number_of_samples

    return logloss_score
