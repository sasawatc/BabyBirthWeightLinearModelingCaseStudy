# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:37:39 2019

@author: Kaiyi

Purpose: team project

working dictonary: C:\Users\Administrator\Desktop\Hult\Machine learning\Project
"""

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

file = 'birthweight.xlsx'
data = pd.read_excel(file)

##############################################################################
# checking for missing value
##############################################################################
print(data.isnull().sum()/data.shape[0])


for col in data:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if data[col].isnull().any():
        data['m_'+col] = data[col].isnull().astype(int)
        

"""
Missing value only in meduc (mother's education), monpre (month prenatal care began),
npvis (otal number of prenatal visits), fage (father's age), feduc (father's education), 
omaps (one minute apgar score), fmaps (five minute apgar score), 
cigs(avg cigarettes per day), drink (avg drinks per week)
"""

# check skewness 
# create a list that store all names for the columns have missing value
missing_list = ['meduc','monpre', 'npvis', 'fage','feduc','omaps','fmaps',
                'cigs','drink']

for miss_col in missing_list:
    sns.distplot(data[miss_col].dropna())
    plt.show()

# data is skewed. fill missing value with mediam
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
mage_limit = 43
meduc_limit = 6
monpre_limit = 3
npvis_limit_high = 15
npvis_limit_low = 6
fage_limit = 43
feduc_limit = 6
omaps_limit = 7

# need more discuss
fmaps_limit_high = 8
cigs_limit = 0
drink_limit = 0

# create new data frame for flagging outlier: out_flag

out_flag = no_missing.copy()

# flag outliers: 1 for higher outliers, -1 for lower outliers
# mage
out_flag['o_mage'] = 0

for index, value in enumerate(out_flag.loc[:, 'mage']):
    if value > mage_limit:
        out_flag.loc[index, 'o_mage'] = 1
    
# meduc
out_flag['o_meduc'] = 0

for index, value in enumerate(out_flag.loc[:, 'meduc']):
    if value < meduc_limit:
        out_flag.loc[index, 'o_meduc'] = -1

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
        
# save to excel
out_flag.to_excel('clean data.xlsx')

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


###############################################################################
# OLS regression analysis
###############################################################################

# full model
full_ols = smf.ols(formula = """bwght ~ out_flag['mage'] +
                                        out_flag['meduc'] +
                                        out_flag['monpre'] +
                                        out_flag['npvis'] +
                                        out_flag['fage'] +
                                        out_flag['feduc'] +
                                        out_flag['fmaps'] +
                                        out_flag['cigs'] +
                                        out_flag['drink'] +
                                        out_flag['mwhte'] +
                                        out_flag['mblck'] +
                                        out_flag['moth'] +
                                        out_flag['fwhte'] +
                                        out_flag['fblck'] +
                                        out_flag['foth'] +
                                        out_flag['m_meduc'] +
                                        out_flag['m_monpre'] +
                                        out_flag['m_npvis'] +
                                        out_flag['m_fage'] +
                                        out_flag['m_feduc'] +
                                        out_flag['m_fmaps'] +
                                        out_flag['m_cigs'] +
                                        out_flag['m_drink'] +
                                        out_flag['o_mage'] +
                                        out_flag['o_meduc'] +
                                        out_flag['o_monpre'] +
                                        out_flag['o_npvis'] +
                                        out_flag['o_fage'] +
                                        out_flag['o_feduc'] +
                                        out_flag['o_omaps'] - 1
                                        """,
                                        data = out_flag)


# Fitting Results
result_full = full_ols.fit()

# Summary Statistics
print(result_full.summary())


# significant model
sig_ols = smf.ols(formula = """bwght ~ out_flag['npvis'] +
                                       out_flag['fage'] +
                                       out_flag['fmaps'] +
                                       out_flag['cigs'] +
                                       out_flag['mwhte'] +
                                       out_flag['mblck'] +
                                       out_flag['moth'] +
                                       out_flag['fwhte'] +
                                       out_flag['fblck'] +
                                       out_flag['foth'] +
                                       out_flag['m_fage'] +
                                       out_flag['m_fmaps'] +
                                       out_flag['o_omaps']
                                       """,
                                       data = out_flag)

# Fitting Results
result_sig = sig_ols.fit()

# Summary Statistics
print(result_sig.summary())


###############################################################################
# KNN classifier analysis
###############################################################################

# create new dataframe for knn analysis      

knn_df = out_flag.copy()

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
linear_x = knn_X.loc[:,['fmaps',
                        'male',
                        'fwhte',
                        'foth',
                        'm_meduc',
                        'm_fage',
                        'm_omaps',
                        'm_cigs',
                        'o_mage',
                        'o_meduc',     
                        'o_fage',
                        'o_omaps']]

# split data into training and testing data
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(linear_x,
                                                                                knn_y,
                                                                                test_size = 0.2,
                                                                                random_state = 78) 

# create regressor
reg = LinearRegression()

# fit regressor
reg.fit(X_train_linear,y_train_linear)

# check accuracy
reg.score(X_test_linear,y_test_linear)
















