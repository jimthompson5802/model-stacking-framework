#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 23:42:47 2018

@author: jim
"""

import yaml
import numpy as np

##############################################################
#                                                            #
#          CUSTOMIZE FOR KAGGLE COMPETITION                  #
#                                                            #
##############################################################

from sklearn.metrics import mean_squared_error
###
#
# function to calculate Kaggle performance metric during CV 
# Must be customized for each competition
#
###
def calculateKaggleMetric(y=None,y_hat=None):
    return np.sqrt(mean_squared_error(y,y_hat))




def formatKaggleSubmission(predictions,model_id):
    # predictions is dataframe containing test data set predictions created
    # by the ModelTrainer()
    # returns submission dataframe with correctly formatted submission data set
    #
    
    #
    # get parameters 
    #
   
    CONFIG = getConfigParameters()
    
    
    submission = predictions[CONFIG['ID_VAR']]\
        .join(predictions[model_id+'_0'].apply(lambda x: np.exp(x)-1))
    submission.columns = CONFIG['KAGGLE_SUBMISSION_HEADERS']
        
    return submission