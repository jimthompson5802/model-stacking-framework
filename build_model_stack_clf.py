# -*- coding: utf-8 -*-

###
# This will invoke required python modules to build feature sets
# and models to create the model stack.
###
#%%
import os
import os.path


from framework.model_stacking import getConfigParameters

#%%
#
# get parameters 
#


CONFIG = getConfigParameters()

ROOT_DIR = CONFIG['ROOT_DIR']
MUNGE_DIR = CONFIG['MUNGE_DIR']
MODELS_DIR = CONFIG['MODEL_DIR']
#%%
#
# Create k-fold specification for training
#
exec(open(os.path.join(ROOT_DIR,MUNGE_DIR,'createCVFolds.py')).read())

#%%
#
# Create feature sets for Level 0
#
exec(open(os.path.join(ROOT_DIR,MUNGE_DIR,'createLevel0BaselineFeatureSet.py')).read())
exec(open(os.path.join(ROOT_DIR,MUNGE_DIR,'createLevel0Features.py')).read())

#%%
#
# Train models on Level 0
#
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L0LOG1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L0NN1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L0RF1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L0XTC1','train_model.py')).read())

#%%
#
# Create feature sets for Level 1
#
exec(open(os.path.join(ROOT_DIR,MUNGE_DIR,'createLevel1Features.py')).read())

#%%
#
# Train models on Level 1
#
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L1RF1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L1NN1','train_model.py')).read())

#%%
#
# Create feature sets for Level 2
#
exec(open(os.path.join(ROOT_DIR,MUNGE_DIR,'createLevel2Features.py')).read())

#%%
#
# Train models on Level 2
#
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L2RF1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,MODELS_DIR,'L2NN1','train_model.py')).read())