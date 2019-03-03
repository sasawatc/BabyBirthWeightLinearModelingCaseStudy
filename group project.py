# -*- coding: utf-8 -*-

# Importing new libraries
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.neighbors import KNeighborsClassifier # KNN for classification
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation


# Importing other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

file = 'birthweight_feature_set.xlsx'
data = pd.read_excel('data/' +file)
data.drop(['omaps', 'fmaps'], axis = 1, inplace = True, errors = 'ignore')

##############################################################################
# checking for missing value
##############################################################################
print(data.isnull().sum()/data.shape[0])


for col in data:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if data[col].isnull().any():
        data['m_' +col] = data[col].isnull().astype(int)
        
print(
      data
      .isnull()
      .sum()
      )

"""
Missing value only in meduc (mother's education), 
                      npvis (total number of prenatal visits), 
                      feduc (father's education)
"""

# check skewness 
# create a list that store all names for the columns have missing value
missing_list = ['meduc', 'npvis','feduc']

for miss_col in missing_list:
    sns.distplot(data[miss_col].dropna())
    plt.show()

"""
data is skewed. fill missing value with mediam
"""

# create new data frame without null value: no_missing
no_missing = data.copy()
for miss_col in missing_list:
    fill = no_missing[miss_col].median()
    no_missing[miss_col] = no_missing[miss_col].fillna(fill)
    
# check if any missing value left

print(no_missing.isnull().any().any())
   
        
##############################################################################
# checking for outlier
##############################################################################

# using boxplot to check outliers

for col in no_missing:
    no_missing.boxplot(column = col,
                       vert = False,
                       meanline = True,
                       showmeans = True)
    plt.title(col)
    plt.show()

for col in no_missing:
    sns.distplot(no_missing[col])
    plt.show()


# set limit for each column
mage_limit = 35 
meduc_limit = 14 # didn't finish college
monpre_limit = 3 # starting checkup in first trimester is safest
npvis_limit_high = 14
npvis_limit_low = 6
fage_limit_high = 45
fage_limit_low = 25
feduc_limit = 14 # didnt finish college
drink_limit = 2 
bwght_limit = 2500


"""
no outliers: meduc (added line based on research), cigs
"""


# create new data frame for flagging outlier: out_flag

out_flag = no_missing.copy()

# flag outliers: 1 for higher outliers, -1 for lower outliers
# mage
out_flag['o_mage'] = 0

for index, value in enumerate(out_flag.loc[:, 'mage']):
    if value >= mage_limit:
        out_flag.loc[index, 'o_mage'] = 1

# meduc
out_flag['o_meduc'] = 0

for index, value in enumerate(out_flag.loc[:, 'meduc']):
    if value <= meduc_limit:
        out_flag.loc[index, 'o_meduc'] = 1
        
# monpre
out_flag['o_monpre'] = 0

for index, value in enumerate(out_flag.loc[:, 'monpre']):
    if value > monpre_limit:
        out_flag.loc[index, 'o_monpre'] = 1        

# npvis
out_flag['o_npvis'] = 0

for index, value in enumerate(out_flag.loc[:, 'npvis']):
    if value > npvis_limit_high:
        out_flag.loc[index, 'o_npvis'] = 1 
    elif value < npvis_limit_low:
        out_flag.loc[index, 'o_npvis'] = -1

# fage
out_flag['o_fage'] = 0

for index, value in enumerate(out_flag.loc[:, 'fage']):
    if value > fage_limit_high:
        out_flag.loc[index, 'o_fage'] = 1 
    elif value < fage_limit_low:
        out_flag.loc[index, 'o_fage'] = -1

# feduc
out_flag['o_feduc'] = 0

for index, value in enumerate(out_flag.loc[:, 'feduc']):
    if value <= feduc_limit:
        out_flag.loc[index, 'o_feduc'] = 1


# drink
out_flag['o_drink'] = 0

for index, value in enumerate(out_flag.loc[:, 'drink']):
    if value > drink_limit:
        out_flag.loc[index, 'o_drink'] = 1 
        
# bwght
out_flag['o_bwght'] = 0

for index, value in enumerate(out_flag.loc[:, 'bwght']):
    if value < bwght_limit:
        out_flag.loc[index, 'o_bwght'] = 1

###############################################################################
# regroup the data
###############################################################################
        
# distinct parents into with or without college degree
"""
New columns: fcol: whether father accepted eduction higher than high school
                   1 for yes, 0 for no
             mcol: whether father accepted eduction higher than high school
                   1 for yes, 0 for no
"""

# copy out_flag to a new data frame: new_column
#new_column = out_flag.copy()
#new_column['fcol'] = 0
#for index, value in enumerate(new_column['feduc']):
#    if value > 12:
#        new_column.loc[index, 'fcol'] = 1
#    
#new_column['mcol'] = 0
#for index, value in enumerate(new_column['meduc']):
#    if value > 12:
#        new_column.loc[index, 'mcol'] = 1
        
# race of the parent 
"""
New columns: fwmw: both father and mother are white
                 1 for yes, 0 for no
             fwmb:  father is  white and mother is black
                 1 for yes, 0 for no
             fwmo: father is  white and mother is other
                 1 for yes, 0 for no
             fbmw: father is black and mother is white
                 1 for yes, 0 for no
             fbmb: both father and mother are black
                 1 for yes, 0 for no
             fbmo: father is black and mother is other
                 1 for yes, 0 for no 
             fomw: father is other and mother is white
                 1 for yes, 0 for no    
             fomb: father is other and mother is black
                 1 for yes, 0 for no     
             fomo: both father and mother are other
                 1 for yes, 0 for no
             checking: check for all data is assigned to one of the columns above                                       
"""
out_flag['fwmw'] = 0
out_flag['fwmb'] = 0
out_flag['fwmo'] = 0
out_flag['fbmw'] = 0
out_flag['fbmb'] = 0
out_flag['fbmo'] = 0
out_flag['fomw'] = 0
out_flag['fomb'] = 0
out_flag['fomo'] = 0
out_flag['checking'] = 0 # check for any missing, will be dropped later

for index in range(len(out_flag)):
    if out_flag.loc[index, 'fwhte'] == 1 and out_flag.loc[index, 'mwhte'] == 1:
        out_flag.loc[index, 'fwmw'] = 1
    elif out_flag.loc[index, 'fwhte'] == 1 and out_flag.loc[index, 'mblck'] == 1:
        out_flag.loc[index, 'fwmb'] = 1
    elif out_flag.loc[index, 'fwhte'] == 1 and out_flag.loc[index, 'moth'] == 1:
        out_flag.loc[index, 'fwmo'] = 1
    elif out_flag.loc[index, 'fblck'] == 1 and out_flag.loc[index, 'mwhte'] == 1:
        out_flag.loc[index, 'fbmw'] = 1
    elif out_flag.loc[index, 'fblck'] == 1 and out_flag.loc[index, 'mblck'] == 1:
        out_flag.loc[index, 'fbmb'] = 1
    elif out_flag.loc[index, 'fblck'] == 1 and out_flag.loc[index, 'moth'] == 1:
        out_flag.loc[index, 'fbmo'] = 1
    elif out_flag.loc[index, 'foth'] == 1 and out_flag.loc[index, 'mwhte'] == 1:
        out_flag.loc[index, 'fomw'] = 1
    elif out_flag.loc[index, 'foth'] == 1 and out_flag.loc[index, 'mblck'] == 1:
        out_flag.loc[index, 'fomb'] = 1
    elif out_flag.loc[index, 'foth'] == 1 and out_flag.loc[index, 'moth'] == 1:
        out_flag.loc[index, 'fomo'] = 1 
    else: 
        out_flag.loc[index, 'checking'] = 1
        
out_flag.drop(['checking'], axis = 1, inplace = True, errors = 'ignore') 


out_flag['lbwght'] = np.log(out_flag['bwght'])

# save to excel
out_flag.to_excel('data/clean data.xlsx')


###############################################################################
# Correlation Analysis
###############################################################################
        
# caculate correlation
df_corr = out_flag.corr().round(2)

# plot correlation
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:18, 1:18]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)

plt.show()


print(df_corr['bwght'].sort_values())


###############################################################################
# OLS regression analysis
###############################################################################

# full model
full_ols = smf.ols(formula = """bwght ~   out_flag['mage']  
                                        + out_flag['meduc']  
                                        + out_flag['monpre']  
                                        + out_flag['npvis']  
                                        + out_flag['fage']  
                                        + out_flag['feduc']  
                                        + out_flag['cigs']  
                                        + out_flag['drink']  
                                        + out_flag['male']  
                                        + out_flag['mwhte']  
                                        + out_flag['mblck']  
                                        + out_flag['moth']  
                                        + out_flag['fwhte']  
                                        + out_flag['fblck']  
                                        + out_flag['foth']   
                                        + out_flag['o_mage']  
                                        + out_flag['o_monpre']  
                                        + out_flag['o_npvis']  
                                        + out_flag['o_fage']  
                                        + out_flag['o_feduc']  
                                        + out_flag['o_drink']
                                        + out_flag['fwmw']
                                        + out_flag['fwmb']
                                        + out_flag['fwmo'] 
                                        + out_flag['fbmw']
                                        + out_flag['fbmb']
                                        + out_flag['fbmo']
                                        + out_flag['fomw']
                                        + out_flag['fomb']
                                        + out_flag['fomo']
                                        """,
                                        data = out_flag)


# Fitting Results
result_full = full_ols.fit()

# Summary Statistics
print(result_full.summary())
print(f"""
Parameters:
{result_full.params.round(2)}

Summary Statistics:
R-Squared:          {result_full.rsquared.round(3)}
Adjusted R-Squared: {result_full.rsquared_adj.round(3)}
""")

# significant model
sig_ols = smf.ols(formula = """bwght ~   out_flag['mage']  
                                        + out_flag['meduc']
                                        + out_flag['feduc']  
                                        + out_flag['cigs']  
                                        + out_flag['drink']  
                                        + out_flag['mwhte']   
                                        + out_flag['fblck']  
                                        + out_flag['foth']   
                                        + out_flag['o_mage']    
                                        + out_flag['o_feduc']  
                                        + out_flag['fwmw']
                                        + out_flag['fwmb']
                                        + out_flag['fwmo'] 
                                        + out_flag['fomb']
                                        """,
                                        data = out_flag)


# Fitting Results
result_sig = sig_ols.fit()

# Summary Statistics
print(result_sig.summary())
print(f"""
Parameters:
{result_full.params.round(2)}

Summary Statistics:
R-Squared:          {result_full.rsquared.round(3)}
Adjusted R-Squared: {result_full.rsquared_adj.round(3)}
""")
# 0.724
out_flag_target = out_flag.loc[:, 'bwght']
out_flag_data = out_flag.loc[:, ['mage',
                                 'meduc',
                                 'feduc',
                                 'cigs',
                                 'drink',
                                 'mwhte',
                                 'fblck',
                                 'foth',
                                 'o_mage',
                                 'o_feduc',
                                 'fwmw',
                                 'fwmb',
                                 'fwmo',
                                 'fomb']]
    
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
            out_flag_data,
            out_flag_target,
            test_size = 0.1,
            random_state = 508)



# Prepping the Model
lr = LinearRegression(fit_intercept = False)
# fit_intercept = false - don't want it cuz then we'll be able to get R square of 0.975

# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{lr_pred.round(2)}
""")


# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score_ols_optimal)
# 0.886 compared to KNeighbors 0.617
# Linear regression works better than KNeighbors-N

# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

#Training Score 0.7502
#Testing Score: 0.6579



#Kathy's notes stop right here!









###############################################################################
# Generalization using Train/Test Split
###############################################################################
# create new dataframe for knn analysis      

knn_df = out_flag.copy()

# prepare features: knn_X and target: knn_y

data_X = knn_df.drop(['bwght', 'o_bwght'], axis = 1)

target_y = knn_df.loc[:,'bwght']

# split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(data_X,
                                                    target_y,
                                                    test_size = 0.2,
                                                    random_state = 508)

###############################################################################
# Forming a Base for Machine Learning with KNN
###############################################################################

########################
# Step 1: Create a model object
########################

knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 1)

# Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)

# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)

# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")

# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)
# basically an R square value


# The score is directly comparable to R-Square
print(y_score)

###############################################################################
# How Many Neighbors?
###############################################################################

# Loop to choose the best neighbor

neighbors = np.arange(1,51)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    """ loop though neighbors to find the accuracy rate for each neighbors for 
    both train and test data """
    
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# plot the accuracy
plt.title('k-NN: varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Test Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Train Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


###############################################################################
# KNN regression analysis
###############################################################################

# choose the best neighbor

neighbors = np.arange(1,30)
train_accuracy_reg = np.empty(len(neighbors))
test_accuracy_reg = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    """ loop though neighbors to find the accuracy rate for each neighbors for 
    both train and test data """
    
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    train_accuracy_reg[i] = knn.score(X_train, y_train)
    test_accuracy_reg[i] = knn.score(X_test, y_test)

# plot the accuracy
plt.title('k-NN Regression: varying Number of Neighbors')
plt.plot(neighbors, test_accuracy_reg, label = 'Test Accuracy')
plt.plot(neighbors, train_accuracy_reg, label = 'Train Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(test_accuracy) 
print(max(test_accuracy))

print(test_accuracy.index(max(test_accuracy)))


# Building a model with k = 5
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 5)

# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)



# Scoring the model
y_score = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score)
# 0.667!


print(f"""
Our base to compare other models is {y_score.round(3)}.
    
This base helps us evaluate more complicated models and lets us consider
tradeoffs between accuracy and interpretability.
""")


###############################################################################
# regression analysis (not KNN)
###############################################################################

from sklearn.linear_model import LinearRegression # import regression model
from sklearn.linear_model import Lasso # import Lasso
# choose important features
lasso = Lasso(alpha = 0.3, normalize = True)

lasso.fit(X_train, y_train)

# compute coefficient
lasso_coef = lasso.coef_

# plot coefficient
column_names = data_X.columns
fig, ax = plt.subplots(figsize=(12,12))
plt.plot(range(len(column_names)), lasso_coef)
plt.xticks(range(len(column_names)), column_names.values, rotation = 60)
plt.margins(0.02)
plt.show()

"""
Important features are the non-zero features on the graph:
positive coefficient:    
                    fbomo,
                    o_drink,
                    fomb,
                    o_npvis,
                    o_mage
negative coefficient:                    
                    foth,
                    m_omaps,
                    m_cigs,
                    o_mage,
                    o_fage,
                    o_omaps
"""
# create new features data frame: linear_x
linear_x = knn_X.loc[:,['meduc',
                        'feduc',
                        'cigs',
                        'drink',
                        'male',
                        'mblck',
                        'foth',
                        'm_meduc',
                        'm_npvis',
                        'm_feduc',     
                        'o_mage',
                        'o_npvis',
                        'o_fage',
                        'o_feduc',
                        'o_omaps',
                        'o_fmaps',
                        'o_drink']]

# split data into training and testing data
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(linear_x,
                                                                                knn_y,
                                                                                test_size = 0.2,
                                                                                random_state = 78) 

# create regressor
reg = LinearRegression(fit_intercept=True, normalize = True)

# fit regressor
reg.fit(X_train_linear,y_train_linear)

# check accuracy
reg.score(X_test_linear,y_test_linear)
















