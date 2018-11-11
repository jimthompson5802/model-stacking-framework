# -*- coding: utf-8 -*-
#%%
from msw.model_stacking import ModelTrainer, ModelPerformanceTracker
from sklearn.neural_network import MLPRegressor as ThisModel

#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(hidden_layer_sizes=(15,),random_state=13), #hyper-parameters
        model_id='L0NN1',   # Model Identifier
        feature_set='KFS01'  # feature set to use
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

