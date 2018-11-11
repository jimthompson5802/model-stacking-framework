from msw.model_stacking import FeatureGenerator, getConfigParameters
from sklearn.preprocessing import MinMaxScaler, Imputer, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


#%%
import pandas as pd



#%%
#
# get parameters 
#

CONFIG = getConfigParameters()

#%%
#
# feature set 1
#
print("Preparing Level 0 Feature Set 1")

comment="""
all attributes in train and test data set.
missing values set to -999
"""

fs = FeatureGenerator('raw','KFS01',comment=comment)

# get raw data
X_train, y_train, X_test = fs.getRawData(train_ds='train.csv', 
                                         test_ds='test.csv')


##############################################################
#                                                            #
#          CUSTOMIZE FOR KAGGLE COMPETITION                  #
#                                                            #
##############################################################

X_train.fillna(-999,inplace=True)

X_test.fillna(-999,inplace=True)


########### END OF KAGGLE COMPETITION CUSTOMIZATION #########

fs.saveFeatureSet(X_train, y_train, X_test)


#%%
#
# feature set 2
#
print("Preparing Level 0 Feature Set 2")
comment="""
Only numeric features in train and test data set.
missing values set to -999
"""

fs = FeatureGenerator('raw','KFS02',comment=comment)

# get raw data
X_train, y_train, X_test = fs.getRawData(train_ds='train.csv', test_ds='test.csv')

##############################################################
#                                                            #
#          CUSTOMIZE FOR KAGGLE COMPETITION                  #
#                                                            #
##############################################################


# find only numberic attributes
numeric_predictors = [x for x in X_train.columns if X_train[x].dtype != 'O']

X_train = X_train.loc[:,numeric_predictors]
X_train.fillna(-999,inplace=True)
X_train.shape

X_test = X_test.loc[:,numeric_predictors]
X_test.fillna(-999,inplace=True)
X_test.shape


########### END OF KAGGLE COMPETITION CUSTOMIZATION #########

fs.saveFeatureSet(X_train, y_train, X_test)

#%%
comment = """
#
# feature set 3 - suitable for nerual network
# one-hot encode categorical variables
# scale numeric to [0,1]
#
"""

fs = FeatureGenerator('raw','KFS03',comment=comment)

# get raw data
X_train, y_train, X_test = fs.getRawData(train_ds='train.csv', test_ds='test.csv')

##############################################################
#                                                            #
#          CUSTOMIZE FOR KAGGLE COMPETITION                  #
#                                                            #
##############################################################

print("Preparing Level 0 Feature Set 3")

# Keep at most this number of most number frequent unique factor levels
TOP_CATEGORICAL_LEVELS = 10

# maximum levels for categorical values
MAX_CATEGORICAL_LEVELS = 100

# Exclude these predictors from the baseline feature set
PREDICTORS_TO_EXCLUDE = []


print('Shape X_train: ',X_train.shape,", Shape X_test:",X_test.shape)

training_rows = X_train.shape[0]


# partition numeric vs categorical predictors
num_predictors = [x for x in X_train.columns if X_train[x].dtype != 'O']

cat_predictors = list(set(X_train.columns) - set(num_predictors))

print('Number of numeric predictors: ',len(num_predictors),', Number of categorical predicators: ',len(cat_predictors))

# one-hot encode categorical predictors
train_encoded_list = []
test_encoded_list = []

for c in cat_predictors:

    # get one column of categorical variables
    train_cat = X_train[c].copy()
    test_cat = X_test[c].copy()
    
    #temporarily combine train and test to get universe of valid values
    all_cat = pd.concat([train_cat,test_cat])
    all_cat.name = train_cat.name
    
    # determine number of unique levels
    number_of_levels = len(all_cat.unique())
    print('Predictor: ',c,' levels ',number_of_levels)
    if number_of_levels > MAX_CATEGORICAL_LEVELS:
        print("    By passing")
        continue
    
    # handle situation where number of unique levels exceed threshold
    if number_of_levels > TOP_CATEGORICAL_LEVELS:
        counts_by_level = all_cat.value_counts()
        
        # get level values for those not in the top ranks
        low_count_levels = counts_by_level.index[TOP_CATEGORICAL_LEVELS:]
    
        # eliminate NULL value if present
        levels_to_other = [x for x in low_count_levels if len(x)>0]
        
        # set less frequent levels to special valid value 
        idx = [x in set(levels_to_other)for x in train_cat]
        train_cat.loc[idx] = '__OTHER__'
        
        idx = [x in set(levels_to_other)for x in test_cat]
        test_cat.loc[idx] = '__OTHER__'
            
    # impute special value for any missing values
    idx = [ isinstance(x,float) for x in all_cat]
    all_cat.loc[idx] = '__N/A__'
    
    # now hot-one encode categorical variable
    lb = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    
    # training categorical attribute
    temp = lb.fit_transform(all_cat)
    temp = temp.reshape(-1,1)
    temp = ohe.fit_transform(temp)
    
    #generate column names for one-hot encoding
    column_names = [all_cat.name + '.' + x for x in lb.classes_]
    
    # split back out to training and test data sets
    train_encoded_list.append(pd.DataFrame(temp[:training_rows],columns=column_names))
    test_encoded_list.append(pd.DataFrame(temp[training_rows:],columns=column_names))
     

# flatten out into single dataframe
X_train_cat = pd.concat(train_encoded_list,axis=1)
X_test_cat = pd.concat(test_encoded_list,axis=1)


# for numeric predictors use median for missing values
X_train_num = X_train.loc[:,num_predictors]
X_test_num = X_test.loc[:,num_predictors]

# impute median value for missing values and scale to [0,1]
imp = Imputer(strategy='median')
mms = MinMaxScaler()

# flag missing values
train_isnan = pd.DataFrame(X_train_num.isnull().astype('int'),index=X_train_num.index)
train_isnan.columns = [c+'_isnan' for c in X_train_num.columns]

# set up missing values in training data
X_train_num = pd.DataFrame(mms.fit_transform(imp.fit_transform(X_train_num)),columns=num_predictors)
X_train_num = pd.concat([X_train_num,train_isnan],axis=1)
X_train_num = X_train_num[sorted(X_train_num.columns)]

# flag missing values
test_isnan = pd.DataFrame(X_test_num.isnull().astype('int'),index=X_test_num.index)
test_isnan.columns = [c+'_isnan' for c in X_test_num.columns]

# wetup missing values in test data
X_test_num = pd.DataFrame(mms.transform(imp.transform(X_test_num)),columns=num_predictors)
X_test_num = pd.concat([X_test_num,test_isnan],axis=1)
X_test_num = X_test_num[sorted(X_test_num.columns)]


# combine numeric and categorical attributes back to new training and test data set
X_train_new = pd.concat([X_train_num,X_train_cat],axis=1)
X_test_new = pd.concat([X_test_num,X_test_cat],axis=1)

########### END OF KAGGLE COMPETITION CUSTOMIZATION #########

# save new feature set
fs.saveFeatureSet(X_train_new, y_train, X_test_new)


#%%
comment="""
#
# feature set 4 - suitable for nerual network
# one-hot encode categorical variables
# Standardize Numeric variables
#
"""

fs = FeatureGenerator('raw','KFS04',comment=comment)

# get raw data
X_train, y_train, X_test = fs.getRawData(train_ds='train.csv', test_ds='test.csv')

##############################################################
#                                                            #
#          CUSTOMIZE FOR KAGGLE COMPETITION                  #
#                                                            #
##############################################################

# Keep at most this number of most number frequent unique factor levels
TOP_CATEGORICAL_LEVELS = 10

# maximum levels for categorical values
MAX_CATEGORICAL_LEVELS = 100

# Exclude these predictors from the baseline feature set
PREDICTORS_TO_EXCLUDE = []

print("Preparing Level 0 Feature Set 4")

print('Shape X_train: ',X_train.shape,", Shape X_test:",X_test.shape)

training_rows = X_train.shape[0]


# partition numeric vs categorical predictors
num_predictors = [x for x in X_train.columns if X_train[x].dtype != 'O']

cat_predictors = list(set(X_train.columns) - set(num_predictors))

print('Number of numeric predictors: ',len(num_predictors),', Number of categorical predicators: ',len(cat_predictors))

# one-hot encode categorical predictors
train_encoded_list = []
test_encoded_list = []

for c in cat_predictors:

    # get one column of categorical variables
    train_cat = X_train[c].copy()
    test_cat = X_test[c].copy()
    
    #temporarily combine train and test to get universe of valid values
    all_cat = pd.concat([train_cat,test_cat])
    all_cat.name = train_cat.name
    
    # determine number of unique levels
    number_of_levels = len(all_cat.unique())
    print('Predictor: ',c,' levels ',number_of_levels)
    if number_of_levels > MAX_CATEGORICAL_LEVELS:
        print("    By passing")
        continue
    
    # handle situation where number of unique levels exceed threshold
    if number_of_levels > TOP_CATEGORICAL_LEVELS:
        counts_by_level = all_cat.value_counts()
        
        # get level values for those not in the top ranks
        low_count_levels = counts_by_level.index[TOP_CATEGORICAL_LEVELS:]
    
        # eliminate NULL value if present
        levels_to_other = [x for x in low_count_levels if len(x)>0]
        
        # set less frequent levels to special valid value 
        idx = [x in set(levels_to_other)for x in train_cat]
        train_cat.loc[idx] = '__OTHER__'
        
        idx = [x in set(levels_to_other)for x in test_cat]
        test_cat.loc[idx] = '__OTHER__'
            
    # impute special value for any missing values
    idx = [ isinstance(x,float) for x in all_cat]
    all_cat.loc[idx] = '__N/A__'
    
    # now hot-one encode categorical variable
    lb = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    
    # training categorical attribute
    temp = lb.fit_transform(all_cat)
    temp = temp.reshape(-1,1)
    temp = ohe.fit_transform(temp)
    
    #generate column names for one-hot encoding
    column_names = [all_cat.name + '.' + x for x in lb.classes_]
    
    # split back out to training and test data sets
    train_encoded_list.append(pd.DataFrame(temp[:training_rows],columns=column_names))
    test_encoded_list.append(pd.DataFrame(temp[training_rows:],columns=column_names))
     

# flatten out into single dataframe
X_train_cat = pd.concat(train_encoded_list,axis=1)
X_test_cat = pd.concat(test_encoded_list,axis=1)


# for numeric predictors use median for missing values
X_train_num = X_train.loc[:,num_predictors]
X_test_num = X_test.loc[:,num_predictors]

# impute median value for missing values and scale to [0,1]
imp = Imputer(strategy='median')
ss = StandardScaler()

# flag missing values
train_isnan = pd.DataFrame(X_train_num.isnull().astype('int'),index=X_train_num.index)
train_isnan.columns = [c+'_isnan' for c in X_train_num.columns]

# set up missing values in training data
X_train_num = pd.DataFrame(mms.fit_transform(imp.fit_transform(X_train_num)),columns=num_predictors)
X_train_num = pd.concat([X_train_num,train_isnan],axis=1)
X_train_num = X_train_num[sorted(X_train_num.columns)]

# flag missing values
test_isnan = pd.DataFrame(X_test_num.isnull().astype('int'),index=X_test_num.index)
test_isnan.columns = [c+'_isnan' for c in X_test_num.columns]

# wetup missing values in test data
X_test_num = pd.DataFrame(mms.transform(imp.transform(X_test_num)),columns=num_predictors)
X_test_num = pd.concat([X_test_num,test_isnan],axis=1)
X_test_num = X_test_num[sorted(X_test_num.columns)]

# combine numeric and categorical attributes back to new training and test data set
X_train_new = pd.concat([X_train_num,X_train_cat],axis=1)
X_test_new = pd.concat([X_test_num,X_test_cat],axis=1)

########### END OF KAGGLE COMPETITION CUSTOMIZATION #########

# save new feature set
fs.saveFeatureSet(X_train_new, y_train, X_test_new)
#%%