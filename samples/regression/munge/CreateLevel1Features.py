#%%
from msw.model_stacking import FeatureGenerator, getConfigParameters

#%%

import os.path 
import pandas as pd
import numpy as np


#%%
#
# get parameters 
#
CONFIG = getConfigParameters()
    
print('root dir: ',CONFIG['ROOT_DIR'])


#%%
#
# Define Feature Generator for Level 1
#
class FeatureGeneratorNextLevel(FeatureGenerator):
    
    #
    # override getRawData() method to handle retrieval of prior
    # model predictions
    #
    
    def getRawData(self):
        features = []
        for model_id in self.in_dir:
            train_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],
                                          self.CONFIG['DATA_DIR'],
                                          model_id,
                                          'train.csv.gz'))

            test_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],
                                          self.CONFIG['DATA_DIR'],
                                          model_id,
                                          'test.csv.gz'))
             
            features.append((train_df,test_df))
            
        return features
    
    
#%%
#
#  Specify location for prior model predictions
#
fs = FeatureGeneratorNextLevel(in_dir=['ML0RF1','ML0NN1'],
                               out_dir='L1FS01')


#%%
#
# retrieve candidate train and test data sets from prior level
#
# this is a list of tuples.  Each tuple structure: (train_df, test_df)
features = fs.getRawData()





#%%

##############################################################
#                                                            #
#          CUSTOMIZE FOR KAGGLE COMPETITION                  #
#                                                            #
##############################################################


#
# Assemble train and test data sets for next Level
#

# assemble train data set
found_first_df = False
for t in features:
    if not found_first_df:
        
        fs.raw_train_id_df = pd.DataFrame(t[0].iloc[:,0])
        y_train_df = pd.DataFrame(t[0].iloc[:,1])
        X_train_df = pd.DataFrame(t[0].iloc[:,2])
        
        fs.raw_test_id_df = pd.DataFrame(t[1].iloc[:,0])
        X_test_df = pd.DataFrame(t[1].iloc[:,1])
        found_first_df = True
    else:
        X_train_df = X_train_df.join(t[0].iloc[:,2])
        X_test_df = X_test_df.join(t[1].iloc[:,1])
        
        
########### END OF KAGGLE COMPETITION CUSTOMIZATION #########

#%%
fs.saveFeatureSet(X_train_df,y_train_df,X_test_df)