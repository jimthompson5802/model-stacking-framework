# -*- coding: utf-8 -*-
#%%
from msw.model_stacking import ModelTrainer, ModelPerformanceTracker
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

#%%
# set up the pipeline
#

ThisModel = Pipeline([
        ('pca',PCA()),
        ('rf',RandomForestRegressor())
        ])
#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(pca__n_components=252,
                          rf__n_estimators=184,
                          rf__n_jobs=-1,
                          rf__random_state=13), #hyper-parameters
        model_id='L0PCA_RF',   # Model Identifier
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

#%%# -*- coding: utf-8 -*-

