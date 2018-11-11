# -*- coding: utf-8 -*-
#%%
from msw.model_stacking import ModelTrainer, ModelPerformanceTracker
from sklearn.ensemble import RandomForestClassifier as ThisModel

#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(n_estimators=800,max_depth=20,n_jobs=-1), #hyper-parameters
        model_id='L2RF1',   # Model Identifier
        feature_set='L2FS01'  # feature set to use
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