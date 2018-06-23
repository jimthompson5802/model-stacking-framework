# -*- coding: utf-8 -*-
###
# This module creates the k-fold specification used to create
# next level features for buidling stacking.
###

#%%
import os
import os.path
import yaml
import pandas as pd
from sklearn.model_selection import KFold
import pickle
from framework.model_stacking import getConfigParameters
#%%
#
# get parameters 
#

CONFIG = getConfigParameters()
    
print('root dir: ',CONFIG['ROOT_DIR'])


#%%
# retreive raw training data
train_df = pd.read_csv(os.path.join(CONFIG['ROOT_DIR'],
                                    CONFIG['DATA_DIR'],'raw','train.csv.gz'))


#%%
# Create the K-Fold specifications for the training data
folds = []

# save index values for each fold.
kf = KFold(n_splits=5,shuffle=True,random_state=29)
for train_index, holdout_index in kf.split(train_df):
    folds.append((train_index,holdout_index))
    

#%%
# save fold specifications
with open(os.path.join(CONFIG['ROOT_DIR'],CONFIG['DATA_DIR'],'k-fold_specification.pkl'),'wb') as f:
    pickle.dump(folds,f)