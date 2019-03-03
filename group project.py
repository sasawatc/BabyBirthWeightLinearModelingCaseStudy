# -*- coding: utf-8 -*-

# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.neighbors import KNeighborsClassifier # KNN for classification
import statsmodels.formula.api as smf # regression modeling
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

##############################################################################
# checking for missing value
##############################################################################
print(data.isnull().sum()/data.shape[0])


for col in data:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if data[col].isnull().any():
        data['m_' +col] = data[col].isnull().astype(int)
        

"""
Missing value only in meduc (mother's education), 
                      npvis (otal number of prenatal visits), 
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

# set limit for each column
mage_limit = 64
monpre_limit = 4
npvis_limit_high = 15
npvis_limit_low = 6
fage_limit = 55
feduc_limit = 6
omaps_limit = 7
fmaps_limit = 9
drink_limit = 12
bwght_limit = 1600


"""
no outliers: meduc, cigs
"""


# create new data frame for flagging outlier: out_flag

out_flag = no_missing.copy()

# flag outliers: 1 for higher outliers, -1 for lower outliers
# mage
out_flag['o_mage'] = 0

for index, value in enumerate(out_flag.loc[:, 'mage']):
    if value > mage_limit:
        out_flag.loc[index, 'o_mage'] = 1

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
    if value > fage_limit:
        out_flag.loc[index, 'o_fage'] = 1

# feduc
out_flag['o_feduc'] = 0

for index, value in enumerate(out_flag.loc[:, 'feduc']):
    if value < feduc_limit:
        out_flag.loc[index, 'o_feduc'] = -1
        
# omaps
out_flag['o_omaps'] = 0

for index, value in enumerate(out_flag.loc[:, 'omaps']):
    if value < omaps_limit:
        out_flag.loc[index, 'o_omaps'] = -1
        
# fmaps
out_flag['o_fmaps'] = 0

for index, value in enumerate(out_flag.loc[:, 'fmaps']):
    if value > fmaps_limit:
        out_flag.loc[index, 'o_fmaps'] = 1 
    elif value < fmaps_limit:
        out_flag.loc[index, 'o_fmaps'] = -1
        
# drink
out_flag['o_drink'] = 0

for index, value in enumerate(out_flag.loc[:, 'drink']):
    if value > drink_limit:
        out_flag.loc[index, 'o_drink'] = 1 
        
# bwght
out_flag['o_bwght'] = 0

for index, value in enumerate(out_flag.loc[:, 'bwght']):
    if value < bwght_limit:
        out_flag.loc[index, 'o_bwght'] = -1

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
new_column = out_flag.copy()
new_column['fcol'] = 0
for index, value in enumerate(new_column['feduc']):
    if value > 12:
        new_column.loc[index, 'fcol'] = 1
    
new_column['mcol'] = 0
for index, value in enumerate(new_column['meduc']):
    if value > 12:
        new_column.loc[index, 'mcol'] = 1
        
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
new_column['fwmw'] = 0
new_column['fwmb'] = 0
new_column['fwmo'] = 0
new_column['fbmw'] = 0
new_column['fbmb'] = 0
new_column['fbmo'] = 0
new_column['fomw'] = 0
new_column['fomb'] = 0
new_column['fomo'] = 0
new_column['checking'] = 0 # check for any missing, will be dropped later

for index in range(len(new_column)):
    if new_column.loc[index, 'fwhte'] == 1 and new_column.loc[index, 'mwhte'] == 1:
        new_column.loc[index, 'fwmw'] = 1
    elif new_column.loc[index, 'fwhte'] == 1 and new_column.loc[index, 'mblck'] == 1:
        new_column.loc[index, 'fwmb'] = 1
    elif new_column.loc[index, 'fwhte'] == 1 and new_column.loc[index, 'moth'] == 1:
        new_column.loc[index, 'fwmo'] = 1
    elif new_column.loc[index, 'fblck'] == 1 and new_column.loc[index, 'mwhte'] == 1:
        new_column.loc[index, 'fbmw'] = 1
    elif new_column.loc[index, 'fblck'] == 1 and new_column.loc[index, 'mblck'] == 1:
        new_column.loc[index, 'fbmb'] = 1
    elif new_column.loc[index, 'fblck'] == 1 and new_column.loc[index, 'moth'] == 1:
        new_column.loc[index, 'fbmo'] = 1
    elif new_column.loc[index, 'foth'] == 1 and new_column.loc[index, 'mwhte'] == 1:
        new_column.loc[index, 'fomw'] = 1
    elif new_column.loc[index, 'foth'] == 1 and new_column.loc[index, 'mblck'] == 1:
        new_column.loc[index, 'fomb'] = 1
    elif new_column.loc[index, 'foth'] == 1 and new_column.loc[index, 'moth'] == 1:
        new_column.loc[index, 'fomo'] = 1 
    else: 
        new_column.loc[index, 'checking'] = 1
        
        
# save to excel
new_column.to_excel('data/clean data.xlsx')

###############################################################################
# Correlation Analysis
###############################################################################
        
# caculate correlation
df_corr = new_column.corr().round(2)

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
full_ols = smf.ols(formula = """bwght ~   new_column['mage']  
                                        + new_column['meduc']  
                                        + new_column['monpre']  
                                        + new_column['npvis']  
                                        + new_column['fage']  
                                        + new_column['feduc']  
                                        + new_column['cigs']  
                                        + new_column['drink']  
                                        + new_column['male']  
                                        + new_column['mwhte']  
                                        + new_column['mblck']  
                                        + new_column['moth']  
                                        + new_column['fwhte']  
                                        + new_column['fblck']  
                                        + new_column['foth']  
                                        + new_column['m_meduc']  
                                        + new_column['m_npvis']  
                                        + new_column['m_feduc']  
                                        + new_column['o_mage']  
                                        + new_column['o_monpre']  
                                        + new_column['o_npvis']  
                                        + new_column['o_fage']  
                                        + new_column['o_feduc']  
                                        + new_column['o_drink']
                                        """,
                                        data = new_column)


# Fitting Results
result_full = full_ols.fit()

# Summary Statistics
print(result_full.summary())


# significant model
sig_ols = smf.ols(formula = """bwght ~  new_column['mage']  
                                       + new_column['cigs']  
                                       + new_column['drink']  
                                       + new_column['mwhte']  
                                       + new_column['mblck']  
                                       + new_column['moth']  
                                       + new_column['fwhte']  
                                       + new_column['fblck']  
                                       + new_column['foth']  
                                       + new_column['m_npvis']
                                       """,
                                       data = new_column)

# Fitting Results
result_sig = sig_ols.fit()

# Summary Statistics
print(result_sig.summary())

try_ols = smf.ols(formula = """bwght ~  new_column['drink']  
                                      + new_column['cigs']  
                                      + new_column['mage']  
                                      + new_column['o_mage']  
                                      + new_column['fage']  
                                      + new_column['o_fage']  
                                      + new_column['o_drink']  
                                      + new_column['m_meduc']  
                                      + new_column['mwhte']  
                                      + new_column['male']  
                                      + new_column['fblck']  
                                      + new_column['mblck']  
                                      + new_column['feduc']  
                                      + new_column['o_feduc']
                                       """,
                                       data = new_column)

# Fitting Results
result_try = try_ols.fit()

# Summary Statistics
print(result_try.summary())


try_ols = smf.ols(formula = """bwght ~  birth_df['drink']  
                                      + birth_df['cigs']  
                                      + birth_df['mage']  
                                      + birth_df['o_mage']  
                                      + birth_df['fage']  
                                      + birth_df['o_fage']  
                                      + birth_df['o_drink']  
                                      + birth_df['m_meduc']  
                                      + birth_df['male']  
                                      + birth_df['feduc']  
                                      + birth_df['o_feduc']  
                                      + birth_df['mrace']  
                                      + birth_df['frace']
                                       """,
                                       data = birth_df)


###############################################################################
# KNN classifier analysis
###############################################################################

# create new dataframe for knn analysis      

knn_df = new_column.copy()

# prepare features: knn_X and target: knn_y

knn_X = knn_df.drop('bwght', axis = 1)

knn_y = knn_df.loc[:,'bwght']

# split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(knn_X,
                                                    knn_y,
                                                    test_size = 0.2,
                                                    random_state = 78)

# choose the best neighbor

neighbors = np.arange(1,30)
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
plt.title('k-NN: varuing Number of Neighbors')
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
plt.title('k-NN Regression: varuing Number of Neighbors')
plt.plot(neighbors, test_accuracy_reg, label = 'Test Accuracy')
plt.plot(neighbors, train_accuracy_reg, label = 'Train Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


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
column_names = knn_X.columns
fig, ax = plt.subplots(figsize=(12,12))
plt.plot(range(len(column_names)), lasso_coef)
plt.xticks(range(len(column_names)), column_names.values, rotation = 60)
plt.margins(0.02)
plt.show()

"""
Important features are the non-zero features on the graph:
positive coefficient:    
                    fmaps,
                    male,
                    fwhte,
                    m_meduc,
                    m_fage,
                    o_meduc
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
















