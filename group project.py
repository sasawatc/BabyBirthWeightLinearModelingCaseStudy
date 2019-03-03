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
                      npvis (total number of prenatal visits), 
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
full_ols = smf.ols(formula="""bwght ~   new_column['mage']  
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
                   data=new_column)

# Fitting Results
result_full = full_ols.fit()

# Summary Statistics
print(result_full.summary())
print(f"""
Parameters:
{result_full.params.round(2)}

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


try_ols = smf.ols(formula="""bwght ~  birth_df['drink']  
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
                  data=birth_df)

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

knn_X = knn_df.drop('bwght', axis=1)

knn_y = knn_df.loc[:, 'bwght']

# split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(knn_X,
                                                    knn_y,
                                                    test_size=0.1,
                                                    random_state=508)
# choose the best neighbor
###############################################################################
# Forming a Base for Machine Learning with KNN
###############################################################################

neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    """ loop though neighbors to find the accuracy rate for each neighbors for 
    both train and test data """

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

    train_accuracy_reg[i] = knn.score(X_train, y_train)
    test_accuracy_reg[i] = knn.score(X_test, y_test)

# plot the accuracy
plt.title('k-NN: varuing Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Test Accuracy')
plt.plot(neighbors, train_accuracy, label='Train Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###############################################################################
# KNN regression analysis
###############################################################################

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
    knn.fit(X_train, y_train)

    train_accuracy_reg[i] = knn.score(X_train, y_train)
    test_accuracy_reg[i] = knn.score(X_test, y_test)

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
linear_x = knn_X.loc[:, ['meduc',
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





