# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer, ModelPerformanceTracker
from xgboost import XGBClassifier as ThisModel

#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(n_estimators=100,n_jobs=6), #hyper-parameters
        model_id='L0XGB1',   # Model Identifier
        feature_set='KFSBSLN'  # feature set to use
        )

model_tracker = ModelPerformanceTracker(model_trainer=this_model)
#%%
#
# clear out old results
#
this_model.cleanPriorResults()

#%%
#
# train model on all the data
#
this_model.trainModel()

#%%
# create Test predictions
this_model.createTestPredictions()

#%%
#
# create Kaggle submission
#
this_model.createKaggleSubmission()

#%%
#
# record model performance metrics
#
model_tracker.recordModelPerformance()

#%%