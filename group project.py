# -*- coding: utf-8 -*-

# Importing new libraries
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.neighbors import KNeighborsRegressor  # KNN for Regression
import statsmodels.formula.api as smf  # regression modeling
import sklearn.metrics  # more metrics for model performance evaluation

# Importing other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

file = 'birthweight_feature_set.xlsx'
data = pd.read_excel('data/' + file)

##############################################################################
# checking for missing value
##############################################################################
print(data.isnull().sum() / data.shape[0])

for col in data:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if data[col].isnull().any():
        data['m_' + col] = data[col].isnull().astype(int)

"""
Missing value only in meduc (mother's education), 
                      npvis (otal number of prenatal visits), 
                      feduc (father's education)
"""

# check skewness 
# create a list that store all names for the columns have missing value
missing_list = ['meduc', 'npvis', 'feduc']

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
    no_missing.boxplot(column=col,
                       vert=False,
                       meanline=True,
                       showmeans=True)
    plt.title(col)
    plt.show()

outlier_list = ['mage','monpre','npvis','fage','drink', 'cigs']
for column in outlier_list:
    sns.scatterplot(data = no_missing,
                 x = column,
                 y = 'bwght')
    plt.title(column)
    plt.show()


# set limit for each column
mage_limit = 64
monpre_limit = 4
npvis_limit_high = 15
npvis_limit_low = 6
fage_limit = 55
omaps_limit = 7
fmaps_limit = 9
drink_limit = 12
bwght_limit = 1600

"""
no outliers: meduc, cigs
the education year for both father and mother will be separate into 2 group later
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
new_column['checking'] = 0  # check for any missing, will be dropped later

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

# drop checking column
new_column = new_column.drop('checking', axis=1)

# new coulmn: mix_col: if the parent come from different race
new_column['mix_col'] = 1
for index in range(len(new_column)):
    if new_column.loc[index, 'fwmw'] == 1:
        new_column.loc[index, 'mix_col'] = 0
    elif new_column.loc[index, 'fbmb'] == 1:
        new_column.loc[index, 'mix_col'] = 0
    elif new_column.loc[index, 'fomo'] == 1:
        new_column.loc[index, 'mix_col'] = 0 




# save to excel
new_column.to_excel('data/clean data.xlsx')

###############################################################################
# Correlation Analysis
###############################################################################

# caculate correlation
df_corr = new_column.corr().round(2)

# plot correlation

fig, ax = plt.subplots(figsize=(15, 15))

df_corr2 = df_corr.iloc[0:18, 0:18]

sns.heatmap(df_corr2,
            cmap='coolwarm',
            square=True,
            annot=True,
            linecolor='black',
            linewidths=0.5)

plt.show()

print(df_corr['bwght'].sort_values())

###############################################################################
# OLS regression analysis
###############################################################################

"""
There are some columns which data is available after the baby is born, we should
not include them in our model due to we can't have the data before baby is due.
The columns are: omaps:   one minute apgar score 
                 fmaps:   five minute apgar score
                 o_omaps: flag for outliers for one minute apgar score 
                 o_fmaps: flag for outliers for five minute apgar score
                 o_bwght: flag for outliers for baby
Besides, no data for fbmw(father is black, mother is white), and fomw(father is
other, mother is white)
We should not include those columns in our model
"""
# create a new data frame: model_data which contain only the data for model
model_data = new_column.copy()
model_data = model_data.drop(['omaps',
                              'fmaps',
                              'o_omaps',
                              'o_fmaps',
                              'o_bwght',
                              'fbmw',
                              'fomw'],
                              axis = 1)


# full model
full_ols = smf.ols(formula="""bwght ~   model_data['mage']  
                                        + model_data['meduc']  
                                        + model_data['monpre']  
                                        + model_data['npvis']  
                                        + model_data['fage']  
                                        + model_data['feduc']  
                                        + model_data['cigs']  
                                        + model_data['drink']  
                                        + model_data['male']  
                                        + model_data['mwhte']  
                                        + model_data['mblck']  
                                        + model_data['moth']  
                                        + model_data['fwhte']  
                                        + model_data['fblck']  
                                        + model_data['foth']  
                                        + model_data['m_meduc']  
                                        + model_data['m_npvis']  
                                        + model_data['m_feduc']  
                                        + model_data['o_mage']  
                                        + model_data['o_monpre']  
                                        + model_data['o_npvis']  
                                        + model_data['o_fage']    
                                        + model_data['o_drink']
                                        + model_data['fcol']
                                        + model_data['mcol']
                                        + model_data['fwmw']
                                        + model_data['fwmb']
                                        + model_data['fwmo']
                                        + model_data['fbmb']
                                        + model_data['fbmo']
                                        + model_data['fomb']
                                        + model_data['fomo']
                                        + model_data['mix_col']
                                        """,
                   data=model_data)

# Fitting Results
result_full = full_ols.fit()

# Summary Statistics
print(result_full.summary())

# significant model
sig_ols = smf.ols(formula="""bwght ~  model_data['mage']  
                                       + model_data['cigs']  
                                       + model_data['drink']
                                       + model_data['mix_col']
                                       """,
                  data=model_data)

# Fitting Results
result_sig = sig_ols.fit()

# Summary Statistics
print(result_sig.summary())

###############################################################################
# KNN regression analysis
###############################################################################
# create new dataframe for knn analysis      

knn_df = model_data.copy()

# prepare features: knn_X and target: knn_y

knn_X = knn_df.drop('bwght', axis=1)

knn_y = knn_df.loc[:, 'bwght']

# split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(knn_X,
                                                    knn_y,
                                                    test_size=0.1,
                                                    random_state=508)
# choose the best neighbor

neighbors = np.arange(1, 30)
train_accuracy_reg = np.empty(len(neighbors))
test_accuracy_reg = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    """ loop though neighbors to find the accuracy rate for each neighbors for 
    both train and test data """

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_accuracy_reg[i] = knn.score(X_train, y_train)
    test_accuracy_reg[i] = knn.score(X_test, y_test)

# plot the accuracy
plt.title('k-NN Regression: varuing Number of Neighbors')
plt.plot(neighbors, test_accuracy_reg, label='Test Accuracy')
plt.plot(neighbors, train_accuracy_reg, label='Train Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# try to modify 
knn_df2 = model_data.copy()

# drop the columns with duplicate information
knn_df2 = knn_df2.drop(['meduc',
                        'feduc',
                        'mwhte',
                        'mblck',
                        'moth',
                        'fwhte',
                        'fblck',
                        'foth',
                        'fwmb',
                        'fwmo',
                        'fbmo',
                        'fomb',
                        'fwmw',
                        'fbmb',
                        'fomo'],
                        axis = 1)
# prepare features: knn_X and target: knn_y

knn_X2 = knn_df2.drop('bwght', axis=1)

knn_y2 = knn_df2.loc[:, 'bwght']

# split data into training and testing data

X_train2, X_test2, y_train2, y_test2 = train_test_split(knn_X2,
                                                    knn_y2,
                                                    test_size=0.1,
                                                    random_state=508)
# choose the best neighbor

neighbors = np.arange(1, 30)
train_accuracy_reg = np.empty(len(neighbors))
test_accuracy_reg = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    """ loop though neighbors to find the accuracy rate for each neighbors for 
    both train and test data """

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train2, y_train2)

    train_accuracy_reg[i] = knn.score(X_train2, y_train2)
    test_accuracy_reg[i] = knn.score(X_test2, y_test2)

# plot the accuracy
plt.title('k-NN Regression: varuing Number of Neighbors')
plt.plot(neighbors, test_accuracy_reg, label='Test Accuracy')
plt.plot(neighbors, train_accuracy_reg, label='Train Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###############################################################################
# regression analysis (not KNN)
###############################################################################

from sklearn.linear_model import LinearRegression  # import regression model
from sklearn.linear_model import Lasso  # import Lasso

# choose important features
lasso = Lasso(alpha=0.3, normalize=True)

lasso.fit(X_train, y_train)

# compute coefficient
lasso_coef = lasso.coef_

# plot coefficient
column_names = knn_X.columns
fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(range(len(column_names)), lasso_coef)
plt.xticks(range(len(column_names)), column_names.values, rotation=60)
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
linear_x = knn_X.loc[:, ['mage',
                         'meduc',
                         'monpre',
                         'cigs',
                         'drink',
                         'fwhte',
                         'm_npvis',
                         'm_feduc',
                         'o_mage',
                         'o_monpre',
                         'o_npvis',
                         'o_fage',
                         'mcol',
                         'o_drink']]

# split data into training and testing data
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(linear_x,
                                                                                knn_y,
                                                                                test_size=0.2,
                                                                                random_state=508)

# create regressor
reg = LinearRegression(fit_intercept=True, normalize=True)

# fit regressor
reg.fit(X_train_linear, y_train_linear)

# check accuracy
reg.score(X_test_linear, y_test_linear)

###############################################################################
# Tree analysis (not KNN)
###############################################################################
# Importing new libraries
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train2, y_train2)

print('Training Score', tree_full.score(X_train2, y_train2).round(4))
print('Testing Score:', tree_full.score(X_test2, y_test).round(4))


# Creating a tree with only two levels.
tree_2 = DecisionTreeRegressor(max_depth = 3,
                               random_state = 508)

tree_2_fit = tree_2.fit(X_train2, y_train2)


print('Training Score', tree_2.score(X_train2, y_train2).round(4))
print('Testing Score:', tree_2.score(X_test2, y_test2).round(4))


dot_data = StringIO()


export_graphviz(decision_tree = tree_2_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = X_train2.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)





