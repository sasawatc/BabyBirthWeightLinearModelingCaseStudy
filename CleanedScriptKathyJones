#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:38:26 2019

@author: Jonas.B,khuyent.yu
"""
"""
| 1        | mage    | mother's age                    |

| 2        | meduc   | mother's educ                   |

| 3        | monpre  | month prenatal care began       |

| 4        | npvis   | total number of prenatal visits |

| 5        | fage    | father's age, years             |

| 6        | feduc   | father's educ, years            |

| 7        | omaps   | one minute apgar score          |

| 8        | fmaps   | five minute apgar score         |

| 9       | cigs    | avg cigarettes per day          |

| 10       | drink   | avg drinks per week             |

| 11       | male    | 1 if baby male                  |

| 12       | mwhte   | 1 if mother white               |

| 13       | mblck   | 1 if mother black               |

| 14       | moth    | 1 if mother is other            |

| 15       | fwhte   | 1 if father white               |

| 16       | fblck   | 1 if father black               |

| 17       | foth    | 1 if father is other            |

| 18       | bwght   | birthweight, grams              |

"""


# Loading Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = 'Desktop/MachineLearning/birthweight.xlsx'

birth = pd.read_excel(file)

########################
# Fundamental Dataset Exploration
########################
# Column names
birth.columns

# Displaying the first rows of the DataFrame
print(birth.head())

# Dimensions of the DataFrame
birth.shape

# Information about each variable
birth.info()

# Descriptive statistics
birth.describe().round(2)

birth.sort_values('bwght', ascending = False)

###############################################################################
# Imputing Missing Values
###############################################################################

print(
      birth
      .isnull()
      .sum()
      )
# checking birth for nulls, and for every null value, you're adding them together

birth.isnull().sum().sum() #chaining, the last .sum() gives u just 1 total sum of null values

#387 missing values

for col in birth:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if birth[col].isnull().any():
        birth['m_'+col] = birth[col].isnull().astype(int)

# Columns with over 10 missing values: meduc (30), npvis(68), feduc(47), cigs(110), drink(115)

# Creating a dropped dataset
df_dropped = birth.dropna()


#############################
# Exploring 'meduc'
#############################

sns.distplot(df_dropped['meduc'])
# 'meduc' looks binomial. Fill with median.

fill = birth['meduc'].median()
birth['meduc'] = birth['meduc'].fillna(fill)


#############################
# Exploring 'npvis'
#############################

sns.distplot(df_dropped['npvis'])
# normal distribution - can inpute with mean

fill = birth['npvis'].mean()

birth['npvis'] = birth['npvis'].fillna(fill)

#############################
# Exploring 'feduc'
#############################

sns.distplot(df_dropped['feduc'])

# 'feduc' looks binomial. Fill with median.
fill = birth['feduc'].median()

birth['feduc'] = birth['feduc'].fillna(fill)

#############################
# Exploring 'cigs'
#############################

sns.distplot(df_dropped['cigs'])

# 'cigs' seems zero inflated. Imputing with zero.
fill = 0

birth['cigs'] = birth['cigs'].fillna(fill)

#############################
# Exploring 'drink'
#############################

sns.distplot(df_dropped['drink'])

# 'drink' seems zero inflated. Imputing with zero.
fill = 0

birth['drink'] = birth['drink'].fillna(fill)

#############################
# Other missing values
#############################

#'monpre'
sns.distplot(df_dropped['monpre']) #bimodal means impute with median
fill = birth['monpre'].median()
birth['monpre'] = birth['monpre'].fillna(fill)

# 'fage'
sns.distplot(df_dropped['fage']) # normal distribution, impute mean
fill = birth['fage'].mean()
birth['fage'] = birth['fage'].fillna(fill)

# 'omaps'
sns.distplot(df_dropped['omaps']) #bimodal means impute with median
fill = birth['omaps'].median()
birth['omaps'] = birth['omaps'].fillna(fill)

#'fmaps'
sns.distplot(df_dropped['fmaps']) # normal distribution, impute mean
fill = birth['fmaps'].mean()
birth['fmaps'] = birth['fmaps'].fillna(fill)


# Checking the overall dataset to see if there are any remaining
# missing values
print(
      birth
      .isnull()
      .any()
      .any()
      )

###############################################################################
# Outlier Analysis
###############################################################################

birth_quantiles = birth.loc[:, :].quantile([0.20,
                                                0.40,
                                                0.60,
                                                0.80,
                                                1.00])

    
print(birth_quantiles)

for col in birth:
    print(col)
    
"""

Assumed Continuous/Interval Variables - 

mage
meduc
monpre
npvis
fage
feduc
omaps
fmaps
cigs
drink
bwght


Assumed Categorical -

male
mwhte
mblck
moth
fwhte
fblck
foth


Binary Classifiers -

m_meduc
m_monpre
m_npvis
m_fage
m_feduc
m_omaps
m_fmaps
m_cigs
m_drink

"""

########################
# Visual EDA (Histograms)
########################


plt.subplot(2, 2, 1)
sns.distplot(birth['mage'],
             bins = 35,
             color = 'g')

plt.xlabel('mage')


########################


plt.subplot(2, 2, 2)
sns.distplot(birth['meduc'],
             bins = 30,
             color = 'y')

plt.xlabel('meduc')



########################


plt.subplot(2, 2, 3)
sns.distplot(birth['monpre'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('monpre')



########################


plt.subplot(2, 2, 4)

sns.distplot(birth['npvis'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')
plt.xlabel('npvis')
plt.tight_layout()
#plt.savefig('birth Data Histograms 1 of 5.png')
plt.show()



########################
########################

plt.subplot(2, 2, 1)
sns.distplot(birth['fage'],
             bins = 35,
             color = 'g')

plt.xlabel('fage')


########################


plt.subplot(2, 2, 2)
sns.distplot(birth['feduc'],
             bins = 30,
             color = 'y')

plt.xlabel('feduc')



########################


plt.subplot(2, 2, 3)
sns.distplot(birth['omaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('omaps')



########################


plt.subplot(2, 2, 4)

sns.distplot(birth['fmaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('fmaps')



plt.tight_layout()
#plt.savefig('birth Data Histograms 2 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birth['cigs'],
             bins = 30,
             color = 'y')

plt.xlabel('cigs')



########################

plt.subplot(2, 2, 2)
sns.distplot(birth['drink'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('drink')



########################

plt.subplot(2, 2, 3)

sns.distplot(birth['male'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('male')



########################

plt.subplot(2, 2, 4)
sns.distplot(birth['mwhte'],
             bins = 35,
             color = 'g')

plt.xlabel('mwhte')



plt.tight_layout()
#plt.savefig('birth Data Histograms 3 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birth['mblck'],
             bins = 30,
             color = 'y')

plt.xlabel('mblck')



########################


plt.subplot(2, 2, 2)
sns.distplot(birth['moth'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('moth')



########################

plt.subplot(2, 2, 3)

sns.distplot(birth['fwhte'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('fwhte')



plt.subplot(2, 2, 4)
sns.distplot(birth['fblck'],
             bins = 35,
             color = 'g')

plt.xlabel('fblck')



plt.tight_layout()
#plt.savefig('birth Data Histograms 4 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)

sns.distplot(birth['foth'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('foth')



plt.subplot(2, 2, 2)

sns.distplot(birth['bwght'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('bwght')



plt.tight_layout()
#plt.savefig('birth Data Histograms 5 of 5.png')

plt.show()



########################
# Tuning and Flagging Outliers
########################

# Outlier flags

mage_hi = 42
mage_lo = 18
meduc_hi = 17
meduc_lo = 11
monpre_hi = 5 # recommended start prenatal care 5 months before birth
npvis_hi = 22 # recommended at least 14 visits for healthy babies
fage_hi = 44
fage_lo = 5
feduc_hi = 17
feduc_lo = 10
omaps_lo = 6 # anything below 6 means low apgar score (baby needs some help)
fmaps_lo = 6 # anything below 6 means low apgar score (baby needs some help)
cigs_hi = 1
drink_hi = 2
bwght_lo = 2000
bwght_hi = 5500

# mage
birth['out_mage'] = 0

for val in enumerate(birth.loc[ : , 'mage']):
    
    if val[1] <= mage_lo:
        birth.loc[val[0], 'out_mage'] = -1

for val in enumerate(birth.loc[ : , 'mage']):
    
    if val[1] >= mage_hi:
        birth.loc[val[0], 'out_mage'] = 1

# meduc
birth['out_meduc'] = 0

for val in enumerate(birth.loc[ : , 'meduc']):
    
    if val[1] <= meduc_lo:
        birth.loc[val[0], 'out_meduc'] = -1

for val in enumerate(birth.loc[ : , 'meduc']):
    
    if val[1] >= meduc_hi:
        birth.loc[val[0], 'out_meduc'] = 1

# monpre
birth['out_monpre'] = 0

for val in enumerate(birth.loc[ : , 'monpre']):
    
    if val[1] >= monpre_hi:
        birth.loc[val[0], 'out_monpre'] = 1
        
# npvis
birth['out_npvis'] = 0

for val in enumerate(birth.loc[ : , 'npvis']):
    
    if val[1] < npvis_hi:
        birth.loc[val[0], 'out_npvis'] = 1
        
# fage
birth['out_fage'] = 0

for val in enumerate(birth.loc[ : , 'fage']):
    
    if val[1] <= fage_lo:
        birth.loc[val[0], 'out_fage'] = -1

for val in enumerate(birth.loc[ : , 'fage']):
    
    if val[1] >= fage_hi:
        birth.loc[val[0], 'out_fage'] = 1
        
# feduc
birth['out_feduc'] = 0

for val in enumerate(birth.loc[ : , 'feduc']):
    
    if val[1] <= feduc_lo:
        birth.loc[val[0], 'out_feduc'] = -1

for val in enumerate(birth.loc[ : , 'feduc']):
    
    if val[1] >= feduc_hi:
        birth.loc[val[0], 'out_feduc'] = 1
        
# omaps
birth['out_omaps'] = 0

for val in enumerate(birth.loc[ : , 'omaps']):
    
    if val[1] <= omaps_lo:
        birth.loc[val[0], 'out_omaps'] = -1

# fmaps
birth['out_fmaps'] = 0

for val in enumerate(birth.loc[ : , 'fmaps']):
    
    if val[1] <= fmaps_lo:
        birth.loc[val[0], 'out_fmaps'] = -1

# cigs
birth['out_cigs'] = 0

for val in enumerate(birth.loc[ : , 'cigs']):
    
    if val[1] >= cigs_hi:
        birth.loc[val[0], 'out_cigs'] = 1

# drink
birth['out_drink'] = 0

for val in enumerate(birth.loc[ : , 'drink']):
    
    if val[1] >= drink_hi:
        birth.loc[val[0], 'out_drink'] = 1

# bwght
birth['out_bwght'] = 0

for val in enumerate(birth.loc[ : , 'bwght']):
    
    if val[1] <= bwght_lo:
        birth.loc[val[0], 'out_bwght'] = -1

for val in enumerate(birth.loc[ : , 'bwght']):
    
    if val[1] >= bwght_hi:
        birth.loc[val[0], 'out_bwght'] = 1

###############################################################################
# Qualitative Variable Analysis (Boxplots)
###############################################################################
        
"""

Assumed Categorical -

| 11       | male    | 1 if baby male                  |

| 12       | mwhte   | 1 if mother white               |

| 13       | mblck   | 1 if mother black               |

| 14       | moth    | 1 if mother is other            |

| 15       | fwhte   | 1 if father white               |

| 16       | fblck   | 1 if father black               |

| 17       | foth    | 1 if father is other            |

"""

########################
# male

birth.boxplot(column = ['bwght'],
                by = ['male'],
                vert = True,
                patch_artist = False,
                meanline = False,
                showmeans = False)


plt.title("birth weight if baby is male")
plt.suptitle("")

#plt.savefig("Birth weight by Male.png")

plt.show()

########################
# Mwhte
birth.boxplot(column = ['bwght'],
                by = ['mwhte'],
                vert = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True)
plt.title("birth weight if mother is white")
plt.suptitle("")
#plt.savefig("birth weight if mother is white.png")
plt.show()

########################
# mblck
birth.boxplot(column = ['bwght'],
                by = ['mblck'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)
plt.title("birth weight if mother is black")
plt.suptitle("")
#plt.savefig("birth weight if mother is black.png")
plt.show()

########################
# moth


birth.boxplot(column = ['bwght'],
                by = ['moth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)
plt.title("birth weight if mother is other")
plt.suptitle("")
#plt.savefig("birth weight if mother is other.png")
plt.show()

########################
# fwhte


birth.boxplot(column = ['bwght'],
                by = ['fwhte'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)
plt.title("birth weight if father is white")
plt.suptitle("")
#plt.savefig("birth weight if father is white.png")
plt.show()

########################
# fblck


birth.boxplot(column = ['bwght'],
                by = ['fblck'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)
plt.title("birth weight if father is black")
plt.suptitle("")
#plt.savefig("birth weight if father is black.png")
plt.show()

########################
# foth


birth.boxplot(column = ['bwght'],
                by = ['foth'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)
plt.title("birth weight if father is other")
plt.suptitle("")
#plt.savefig("birth weight if father is other.png")
plt.show()

###############################################################################
# Correlation Analysis
###############################################################################

birth.head()


df_corr = birth.corr().round(2)


print(df_corr)


df_corr.loc['bwght'].sort_values(ascending = False)

"""
########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

birth2 = birth.iloc[1:19, 1:19]

sns.heatmap(birth,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


#plt.savefig('birth Correlation Heatmap.png')
plt.show()

"""
#birth.to_excel('Birth_explored.xlsx')

#clean_file = 'Birth_explored.xlsx'

#birth_df = pd.read_excel(clean_file)

###############################################################################
# Univariate Regression Analysis
###############################################################################
import statsmodels.formula.api as smf

birth_data   = birth.loc[:,['fmaps','omaps','out_fmaps','out_omaps','npvis','fage',
                        'male','fwhte','mwhte','feduc','meduc']]

birth_target = birth.loc[:,'bwght']

lm_significant = smf.ols(formula = """bwght ~ birth['fmaps']+
                                             birth['omaps'] +
                                             birth['out_fmaps'] +
                                             birth['out_omaps'] +
                                             birth['npvis'] +
                                             birth['fage'] +
                                             birth['male'] +
                                             birth['fwhte'] +
                                             birth['mwhte'] +
                                             birth['feduc'] +
                                             birth['meduc']
                                             -1
                                             """,
                                                  data = birth)
# Fitting Results
results = lm_significant.fit()

# Printing Summary Statistics
print(results.summary())






birth_concat = pd.concat([X_train, y_train], axis = 1)
# Review of statsmodels.ols
# Step 1: Build the model
lm_significant = smf.ols(formula = """bwght ~ birth_concat['fmaps']+
                                             birth_concat['omaps'] +
                                             birth_concat['out_fmaps'] +
                                             birth_concat['out_omaps'] +
                                             birth_concat['npvis'] +
                                             birth_concat['fage'] +
                                             birth_concat['male'] +
                                             birth_concat['fwhte'] +
                                             birth_concat['mwhte'] +
                                             birth_concat['feduc'] +
                                             birth_concat['meduc']
                                             -1
                                             """,
                                                  data = birth_concat)
# Fitting Results
results = lm_significant.fit()

# Printing Summary Statistics
print(results.summary())





