
import yaml
import pandas as pd
import os
import os.path
import shutil
import pickle
import datetime
import time
import numpy as np

#from kaggle_user_functions import calculateKaggleMetric, formatKaggleSubmission
__version__ = '0.2.0'

_loaded_config_file = False


#
# Common routine to extract configuration parameters
# Sequence for finding configuration file
#   1. Environment variable MSW_CONFIG_FILE
#   2. 'config.yml' in current working directory
#
def getConfigParameters():
    global _loaded_config_file
    
    try:
        file_name = os.environ['MSW_CONFIG_FILE']
        print("retrieving config file from MSW_CONFIG_FILE environment variable")
    except KeyError:
        print("Retrieving config.yml file from current working directory")
        file_name = 'config.yml'
         
    if not _loaded_config_file:
        print('retrieving parameters from configuration file: {}'.format(file_name))
        _loaded_config_file = True
        
    
    with open(file_name,'r') as f:
        CONFIG = yaml.load(f.read())
        
    return CONFIG



#
# locate user specified Kaggle competition specifiction fuctions
#    calculateKaggleMetric: calculates contest metric from (y and y_hat)
#    formatKaggleSubmission: creates Kaggle submission from test data set predicitons
#

CONFIG = getConfigParameters()
kaggle_functions_file = CONFIG['KAGGLE_CONTEST_FUNCTIONS']
with open(os.path.join(CONFIG['ROOT_DIR'],kaggle_functions_file),'r') as g:
    exec(g.read())



###
#
# Class for generating feature sets
#
###
class FeatureGenerator():
    """
    Methods useful for building feature sets for modeling.  

    Expected usage:
        
    fs = FeatureGenrator()
    
    X_train, y_train, X_test = fs.getRawData()
    
    # user specifed code to create feature set and perform tasks such as
    #    imputing missing values
    #    encoding categorical values
    #    standardize values
    #    create synthetic variables
    X_train_fs = ...
    y_train_fs = ...
    X_test_fs =
    
    fs.saveFeatureSet(X_train_fs, y_train_fs, X_test_fs)
    
    Arguments:
        in_dir: directory containing the input data set to transform to a 
                feature set.
        out_dir: directory to contain the new feature set.

    """    
    
    def __init__(self,
                 in_dir=None,  # directory containing input training/test data sets
                 out_dir=None, # directory to contain generated feature set
                 compress_feature_set=True,
                 comment=""
                 ):

        self.in_dir = in_dir        
        self.out_dir = out_dir
        self.compress_feature_set = compress_feature_set
        self.comment = comment
        self.__version__ = __version__
        
        #
        # get parameters 
        #
        
        self.CONFIG = getConfigParameters()
            
        self.root_dir = self.CONFIG['ROOT_DIR']
        self.data_dir = self.CONFIG['DATA_DIR']
        self.id_vars = self.CONFIG['ID_VAR']      
        self.target_var = self.CONFIG['TARGET_VAR'] 
        self.compress_feature_set = compress_feature_set
        self.comment = comment
        
        
        self._makeOutputDirectory()
       
    def _makeOutputDirectory(self):
        #create directory to hold feature set
        # clean out out_dir
        try:
            shutil.rmtree(os.path.join(self.root_dir,self.data_dir,self.out_dir))
        except:
            pass
        
        os.makedirs(os.path.join(self.root_dir,self.data_dir,self.out_dir))
    
    
    def getRawData(self,train_ds='train.csv.gz',test_ds='test.csv.gz'):
        """
        default behaviour - can be overridden for different raw data structures

        Assumes existence of train.csv and test.csv in in_dir location
        Expected function: create raw_id_df, raw_target_df, raw_train_features_df
        and raw_test_features_df data frames.
        
        Returns tuple containing the following elements
            raw_train_features_df: dataframe containing only train predictors to transform
            raw_train_target_df: dataframe containing target variable transform
            raw_test_features_df: dataframe containing test predictors to transform
        """
        df = pd.read_csv(os.path.join(self.root_dir,self.data_dir,self.in_dir,train_ds))
        
        # split data into identifiers, predictors and target data frames
        self.raw_train_id_df = df.loc[:,self.id_vars]
        raw_train_target_df = df.loc[:,[self.target_var]]
        
        # isolate predictor variables
        predictors = sorted(set(df.columns) - set(self.id_vars) - set([self.target_var]))
        
        # isoloate training predictiors
        raw_train_features_df = df.loc[:,predictors]
        
        # get test data set
        df = pd.read_csv(os.path.join(self.root_dir,self.data_dir,self.in_dir,test_ds))
        self.raw_test_id_df = df.loc[:,self.id_vars]
        raw_test_features_df = df.loc[:,predictors]
        
        return raw_train_features_df, raw_train_target_df, raw_test_features_df
        
    
    def saveFeatureSet(self,new_train_features_df=None,
                       new_train_target_df=None,
                       new_test_features_df=None):
        """
        default behaviour - can be overriddent for different new feature storage
        
        append id_vars and target_var save new_train_features_df and 
        new_test_features_df in self.out_dir
        
        append id_vars and target to new feature set and save as csv
        
        Arguments:
            new_train_features_df: dataframe containg transformed train predictors
            new_train_target_df:  dataframe containing transformed target variable
            new_test_features_df: dataframe containing tarnsformed test predictors
        """
        
        if self.compress_feature_set:
            train_ds = 'train.csv.gz'
            test_ds = 'test.csv.gz'
            compression_parm = 'gzip'
        else:
            train_ds = 'train.csv'
            test_ds = 'test.csv'
            compression_parm = None
        

        self.raw_train_id_df.join(new_train_target_df)\
            .join(new_train_features_df)\
            .sort_values(self.id_vars)\
            .to_csv(os.path.join(self.root_dir,self.data_dir,self.out_dir,train_ds),
                    index=False,compression=compression_parm)
        
        self.raw_test_id_df\
            .join(new_test_features_df)\
            .sort_values(self.id_vars)\
            .to_csv(os.path.join(self.root_dir,self.data_dir,self.out_dir,test_ds),
                    index=False,compression=compression_parm)
            
        if len(self.comment) > 0:
            with open(os.path.join(self.root_dir,self.data_dir,self.out_dir,'README.txt'),'w') as f:
                f.write(self.comment)
 
###
#
#  Class for training models
#
###  
class ModelTrainer():
    """
    Methods to train model, convert predictions from the model into features
    for the next level.
    
    Arguments:
        ModelClass: Python class implementation of the model algorithm.  Currently
                    this class is assumed to following class structure of 
                    scikit-learn.
        model_params: Python dictionary specifying parameters for the model.
        model_id: character string used to identify the model
        test_data_method: method for generating test data predictions
                          None - default, retrieve value from config.yml
                          all_data_model - train model on all data to generate test data prediction
                          k-fold_average_model - average test data predictions from k-fold training
        feature_set: Feature identifier used in FeatureGenerator
        train_ds: training data set
        test_ds: test data set
    
    """
    
    def __init__(self,
                 ModelClass=None,  #Model Algorithm
                 model_params={},  # Model hyper-parameters
                 model_id=None,    # model identifier
                 test_prediction_method=None, #training method
                 feature_set=None,  # feature set to use
                 train_ds='train.csv.gz',  # feature set training data set
                 test_ds='test.csv.gz',  # feature set test data set
                 compress_output=True  # compress genrated output
                 ):   
        
        self.ModelClass = ModelClass
        self.model_params = model_params
        self.model_id = model_id
        self.test_prediction_method = test_prediction_method
        self.feature_set = feature_set
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.compress_output = compress_output
        self.out_dir = "M"+model_id
        self.__version__ = __version__
        
        if self.compress_output:
            self.out_train_ds = 'train.csv.gz'
            self.out_test_ds = 'test.csv.gz'
            self.compression_parm = 'gzip'
        else:
            self.out_train_ds = 'train.csv'
            self.out_test_ds = 'test.csv'
            self.compression_parm = None
            
        
        # this implment fix to Issue #1
        self.max_bytes = 2**31 - 1
        
        #
        # get global parameters 
        #
        self.CONFIG = getConfigParameters()
            
        self.root_dir = self.CONFIG['ROOT_DIR']
        self.data_dir = self.CONFIG['DATA_DIR']
        self.model_dir = self.CONFIG['MODEL_DIR']
        
        if self.test_prediction_method == None:
            self.test_prediction_method = self.CONFIG['TEST_PREDICTION_METHOD']
        elif test_prediction_method == 'all_data_model'\
                or test_prediction_method == 'k-fold_average_model':
            self.test_prediction_method = test_prediction_method
        else:
            raise ValueError("test_prediction_method=" + test_prediction_method 
                             + ", valid vaules are 'all_data_model' or 'k-fold_average_model'")       
            
        print('Model training starting for {} with feature set {} at {:%Y-%m-%d %H:%M:%S}'\
              .format(self.model_id,self.feature_set,datetime.datetime.now()))
        print('test_prediction_method: {}'.format(self.test_prediction_method))
        
     
    # added to fix Issue #1
    def _saveModelToDisk(self,model=None):
        # convert to byte stream
        bytes_out = pickle.dumps(model)
        
        # write out byte stream in chunks of size self.max_bytes
        with open(os.path.join(self.root_dir,self.model_dir,
                               self.model_id,self.model_id+'_model.pkl'),'wb') as f:
            for idx in range(0, len(bytes_out), self.max_bytes):
                f.write(bytes_out[idx:idx+self.max_bytes])

    # added to fix Issue #1
    def _loadModelFromDisk(self):
        
        # initailize area to recieve file chunks
        bytes_in = bytearray(0)
        
        # get total size of saved model file
        model_file_name = os.path.join(self.root_dir,self.model_dir,self.model_id,
                               self.model_id+'_model.pkl')
        
        input_size = os.path.getsize(model_file_name)
        
        # read in saved model in max_byte chunks
        with open(model_file_name,'rb') as f:
            for _ in range(0, input_size, self.max_bytes):
                bytes_in += f.read(self.max_bytes) 
        
        # recreate model object
        model = pickle.loads(bytes_in)   
        
        return model
        
            
    def cleanPriorResults(self):
        
        # remove old 
        try:
            shutil.rmtree(os.path.join(self.root_dir,self.data_dir,self.out_dir))
        except:
            pass
        
        os.makedirs(os.path.join(self.root_dir,self.data_dir,self.out_dir))
        
        try:
            os.remove(os.path.join(self.root_dir,self.model_dir,
                                   self.model_id,
                                   self.model_id+'_model.pkl'))
        except:
            pass
        
        try:
            os.remove(os.path.join(self.root_dir,self.model_dir,
                                   self.model_id,
                                   self.model_id+'_submission.csv'))
        except:
            pass
        
            

    def trainModel(self):
        
        print('Starting model training: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        start_training = time.time()
        
        #
        # retrieve KFold specifiction
        #
        with open(os.path.join(self.root_dir,self.data_dir,'k-fold_specification.pkl'),'rb') as f:
            k_folds = pickle.load(f)
            
         
        #
        # generate features for next level
        #
        
        # retrieve training data
        train_df = pd.read_csv(os.path.join(self.root_dir,self.data_dir,
                                            self.feature_set,self.train_ds))
        
        predictors = sorted(list(set(train_df.columns) - 
                                 set(self.CONFIG['ID_VAR']) - set([self.CONFIG['TARGET_VAR']])))
        

        
        
        #
        # create features for next level using the hold out set
        #
        self.cv_performance_metric = []
        
        models_list = []
        next_level = []
        i = 0
        for fold in k_folds:
            i += 1
            print('running fold: {:d} at {:%Y-%m-%d %H:%M:%S}'.format(i,datetime.datetime.now()))
            train_idx = fold[0]
            X_train = train_df.iloc[train_idx,:]
            X_train = X_train.loc[:,predictors]
            y_train = train_df[self.CONFIG['TARGET_VAR']].iloc[train_idx]
            
            model = self.ModelClass(**self.model_params)
            
            model.fit(X_train,y_train)
            
            #generate feature for next level
            # get indices for hold out set
            holdout_idx = fold[1]
            
            # set up predictors and target for hold out set
            X_holdout = train_df.iloc[holdout_idx,:]
            id_holdout = X_holdout.loc[:,self.CONFIG['ID_VAR']]
            X_holdout = X_holdout.loc[:,predictors]
            y_holdout = train_df[self.CONFIG['TARGET_VAR']].iloc[holdout_idx]
            
            # make preduction on hold out set to calculate metric and generate
            # features for next level of stack
            if self.CONFIG['PROBLEM_TYPE'] == 'classification':
                y_hat = model.predict_proba(X_holdout)
            elif self.CONFIG['PROBLEM_TYPE'] == 'regression':
                y_hat = model.predict(X_holdout).reshape(-1,1)
            else:
               raise ValueError("Invalid value for configuration parameter PROBLEM_TYPE: " 
                                + self.CONFIG['PROBLEM_TYPE'] 
                             + ", valid vaules are 'classification' or 'regression'")  
                
            self.cv_performance_metric.append(calculateKaggleMetric(y_holdout,y_hat))
        
            # geneate features for next level
            y_hat = pd.DataFrame(y_hat,index=id_holdout.index)
            y_hat.columns = [self.model_id+'_'+str(col) for col in y_hat.columns]
            y_hat = id_holdout.join(y_holdout).join(y_hat)
            
            next_level.append(y_hat)
            
            if self.test_prediction_method == 'k-fold_average_model':
                models_list.append(model)
            
            
        #
        # combine the generated features into single dataframe & save to disk
        #
        pd.concat(next_level).sort_values(self.CONFIG['ID_VAR'])\
            .to_csv(os.path.join(self.root_dir,self.data_dir,
                                 self.out_dir,
                                 self.out_train_ds),
                    index=False, compression=self.compression_parm)
       
        # method for handling test data
        if self.test_prediction_method == 'all_data_model':
            #
            # train model on complete training data set
            #
            
            print("test_prediction_method='all_data_model', building final model with all data")
    
            X_train = train_df[predictors]
            y_train = train_df[self.CONFIG['TARGET_VAR']]
            
            self.training_rows = X_train.shape[0]
            self.training_columns = X_train.shape[1]
                
            model = self.ModelClass(**self.model_params)
            

            model.fit(X_train,y_train)

            
            self._saveModelToDisk(model)
            
        else:
            self.training_rows = train_df.shape[0]
            self.training_columns = len(predictors)
            self._saveModelToDisk(models_list)        
        
        self.training_time = time.time() - start_training

    def createTestPredictions(self,test_ds='test.csv.gz'):
        #
        # create Kaggle Submission
        #
        # Assumes:  trained model has been saved under "`model_id`_model.pkl"
        #
        
        print('Starting createTestPredictions: {:%Y-%m-%d %H:%M:%S}'\
              .format(datetime.datetime.now()))
        
        model = self._loadModelFromDisk()

        # create data set to make predictions
        test_df = pd.read_csv(os.path.join(self.root_dir,self.data_dir,
                                           self.feature_set,self.test_ds))
        
        predictors = sorted(list(set(test_df.columns) - 
                                 set(self.CONFIG['ID_VAR']) - set([self.CONFIG['TARGET_VAR']])))
        
        test_id = test_df[self.CONFIG['ID_VAR']]
        
        # if single model, then generate predictions for test data
        # if list then generate predictions for each model in list and average test data prediction
        if isinstance(model,(list)):
            print('using k-fold_average_model predictions')
            pred_list = []
            for m in model:
                if self.CONFIG['PROBLEM_TYPE'] == 'classification':
                    y_hat = m.predict_proba(test_df[predictors])
                elif self.CONFIG['PROBLEM_TYPE'] == 'regression':
                    y_hat = m.predict(test_df[predictors]).reshape(-1,1)
                else:
                   raise ValueError("Invalid value for configuration parameter PROBLEM_TYPE: " 
                                    + self.CONFIG['PROBLEM_TYPE'] 
                                 + ", valid vaules are 'classification' or 'regression'")  
                   
                pred_list.append(y_hat)
            
            preds = np.dstack(pred_list).mean(axis=2)
            predictions = pd.DataFrame(preds,index=test_df.index)
                
        else:
            print('using all_data_model predictions for test data set')
            if self.CONFIG['PROBLEM_TYPE'] == 'classification':
                y_hat = model.predict_proba(test_df[predictors])
            elif self.CONFIG['PROBLEM_TYPE'] == 'regression':
                y_hat = model.predict(test_df[predictors]).reshape(-1,1)
            else:
               raise ValueError("Invalid value for configuration parameter PROBLEM_TYPE: " 
                                + self.CONFIG['PROBLEM_TYPE'] 
                             + ", valid vaules are 'classification' or 'regression'")        
               
            predictions = pd.DataFrame(y_hat,index=test_df.index)
            
        predictions.columns = [self.model_id+'_'+str(x) for x in list(predictions.columns)]
        

        # save test predictions for next level
        pred_df = test_id.join(predictions).sort_values(self.CONFIG['ID_VAR'])
        pred_df.to_csv(os.path.join(self.root_dir,self.data_dir,
                                    self.out_dir,
                                    self.out_test_ds), 
               index=False,compression=self.compression_parm)
 


    def createKaggleSubmission(self):
        print('Starting createKaggleSubmission: {:%Y-%m-%d %H:%M:%S}'\
              .format(datetime.datetime.now()))
        
        # retrieve test predictions
        predictions = pd.read_csv(os.path.join(self.root_dir,self.data_dir,
                                    self.out_dir,
                                    self.out_test_ds))
        
        submission = formatKaggleSubmission(predictions,self.model_id)

        
        submission.to_csv(os.path.join(self.root_dir,self.model_dir,
                                       self.model_id,
                                       self.model_id+'_submission.csv'),index=False)
        
        print('Completed createKaggleSubmission: {:%Y-%m-%d %H:%M:%S}'\
              .format(datetime.datetime.now()))
    
###
#
#  Model Performance Tracker
#
###
class ModelPerformanceTracker():
    
    tracking_file = None
    
    def __init__(self,model_trainer=None):
        self.model_trainer = model_trainer
        self.__version__ = __version__
        
        #
        # get global parameters 
        #
        self.CONFIG = getConfigParameters()
            
        self.tracking_file = os.path.join(self.CONFIG['ROOT_DIR'],CONFIG['RESULTS_DIR'],'model_performance_data.csv')
        
        
    
    def recordModelPerformance(self,
                               cv_metric_list=None  # list of cv performance metrics
                               ):
        # retrieve basic model information from model trainer
        model_params = "{'model_params': " + str(self.model_trainer.model_params) \
                + ", 'test_prediction_method': '" + self.model_trainer.test_prediction_method \
                + "'}"
        
        model_id = self.model_trainer.model_id
        feature_set = self.model_trainer.feature_set
        date_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        
        #create a row
        df = pd.DataFrame([date_time,
                           model_id,
                           feature_set,
                           self.model_trainer.training_rows,
                           self.model_trainer.training_columns,
                           self.model_trainer.training_time,
                           np.min(self.model_trainer.cv_performance_metric),   #cv_min_metric
                           np.max(self.model_trainer.cv_performance_metric),   #cv_max_metric
                           np.mean(self.model_trainer.cv_performance_metric),   #cv_avg_metric
                           "",  #public_leaderboard
                           model_params]).T
        df.columns = ['date_time',
                      'model_id',
                      'feature_set',
                      'number_of_rows',
                      'number_of_columns',
                      'training_time',
                      'cv_min_metric',
                      'cv_max_metric',
                      'cv_avg_metric',
                      'public_leaderboard',
                      'model_params']
        
        
        # write out model performance metric
        if not os.path.isfile(self.tracking_file):
            
           df.to_csv(self.tracking_file, header=True, index=False)
           
        else: # else it exists so append without writing the header
        
           df.to_csv(self.tracking_file, mode='a', header=False, index=False)
        
