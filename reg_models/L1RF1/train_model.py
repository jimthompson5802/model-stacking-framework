# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer, ModelPerformanceTracker
from sklearn.ensemble import RandomForestClassifier as ThisModel

#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(n_estimators=200,max_depth=5,n_jobs=-1), #hyper-parameters
        model_id='L1RF1',   # Model Identifier
        feature_set='L1FS01'  # feature set to use
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