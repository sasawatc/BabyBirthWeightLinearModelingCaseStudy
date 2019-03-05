# Standard library imports
from pathlib import Path

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')

data_folder = Path('data/')
output_folder = Path('output/')

data = pd.read_excel(data_folder / 'birthweight_feature_set.xlsx')

##############################################################################
# Checking and imputing missing value
##############################################################################
print(data.isnull().sum() / data.shape[0])

# Create columns: 0 = not missing; 1 = missing
for col in data:
    if data[col].isnull().any():
        data['m_' + col] = data[col].isnull().astype(int)

# Missing value only in meduc (mother's education),
#                       npvis (total number of prenatal visits),
#                       feduc (father's education)

# Check skewness
missing_list = ['meduc', 'npvis', 'feduc']  # columns with missing value

for miss_col in missing_list:
    sns.distplot(data[miss_col].dropna())
    plt.show()

# Data is skewed, fill missing value with median
no_missing = data.copy()

for miss_col in missing_list:
    fill = no_missing[miss_col].median()
    no_missing[miss_col] = no_missing[miss_col].fillna(fill)

# Verify if any missing value left
print(no_missing.isnull().any().any())

##############################################################################
# Checking for outliers
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

##############################################################################
# Flagging for outliers
##############################################################################

# set limit for each column
mage_limit = 35
meduc_limit = 14  # didn't finish college
monpre_limit = 3  # starting checkup in first trimester is safest
npvis_limit_high = 14
npvis_limit_low = 6
fage_limit_high = 45
fage_limit_low = 25
feduc_limit = 14  # didnt finish college
drink_limit = 2
bwght_limit = 2500
# no outliers: meduc (added line based on research), cigs


# create new data frame for flagging outlier: out_flag
out_flag = no_missing.copy()

# Father's Age
out_flag['o_fage'] = 0
for index, value in enumerate(out_flag.loc[:, 'fage']):
    if value > fage_limit_high:
        out_flag.loc[index, 'o_fage'] = 1
    elif value < fage_limit_low:
        out_flag.loc[index, 'o_fage'] = -1

# Father's Education
out_flag['o_feduc'] = 0
for index, value in enumerate(out_flag.loc[:, 'feduc']):
    if value < feduc_limit:
        out_flag.loc[index, 'o_feduc'] = -1

# Mother's Age
out_flag['o_mage'] = 0
for index, value in enumerate(out_flag.loc[:, 'mage']):
    if value >= mage_limit:
        out_flag.loc[index, 'o_mage'] = 1

# Mother's Education (years)
out_flag['o_meduc'] = 0
for index, value in enumerate(out_flag.loc[:, 'meduc']):
    if value <= meduc_limit:
        out_flag.loc[index, 'o_meduc'] = 1

# Month Prenatal Care Began
out_flag['o_monpre'] = 0
for index, value in enumerate(out_flag.loc[:, 'monpre']):
    if value > monpre_limit:
        out_flag.loc[index, 'o_monpre'] = 1

# Total Number of Prenatal Visits
out_flag['o_npvis'] = 0
for index, value in enumerate(out_flag.loc[:, 'npvis']):
    if value > npvis_limit_high:
        out_flag.loc[index, 'o_npvis'] = 1
    elif value < npvis_limit_low:
        out_flag.loc[index, 'o_npvis'] = -1

# AVG Drinks per Week
out_flag['o_drink'] = 0
for index, value in enumerate(out_flag.loc[:, 'drink']):
    if value > drink_limit:
        out_flag.loc[index, 'o_drink'] = 1

# Birth Weight (grams)
out_flag['o_bwght'] = 0
for index, value in enumerate(out_flag.loc[:, 'bwght']):
    if value < bwght_limit:
        out_flag.loc[index, 'o_bwght'] = 1

###############################################################################
# Regroup the data
###############################################################################

# distinct parents into with or without college degree
# New columns: fcol: whether father accepted eduction higher than high school
#                    1 for yes, 0 for no
#              mcol: whether father accepted eduction higher than high school
#                    1 for yes, 0 for no


# race of the parent
# Yes: 1; No: 0
# New columns: fwmw: both father and mother are white
#              fwmb:  father is white and mother is black
#              fwmo: father is white and mother is other
#              fbmw: father is black and mother is white
#              fbmb: both father and mother are black
#              fbmo: father is black and mother is other
#              fomw: father is other and mother is white
#              fomb: father is other and mother is black
#              fomo: both father and mother are other
#              checking: check all data assigned to one of the columns above

out_flag['fwmw'] = 0
out_flag['fwmb'] = 0
out_flag['fwmo'] = 0
out_flag['fbmw'] = 0
out_flag['fbmb'] = 0
out_flag['fbmo'] = 0
out_flag['fomw'] = 0
out_flag['fomb'] = 0
out_flag['fomo'] = 0
out_flag['checking'] = 0  # check for any missing, will be dropped later

for index in range(len(out_flag)):
    if out_flag.loc[index, 'fwhte'] == 1 and out_flag.loc[index, 'mwhte'] == 1:
        out_flag.loc[index, 'fwmw'] = 1
    elif out_flag.loc[index, 'fwhte'] == 1 and out_flag.loc[
        index, 'mblck'] == 1:
        out_flag.loc[index, 'fwmb'] = 1
    elif out_flag.loc[index, 'fwhte'] == 1 and out_flag.loc[index, 'moth'] == 1:
        out_flag.loc[index, 'fwmo'] = 1
    elif out_flag.loc[index, 'fblck'] == 1 and out_flag.loc[
        index, 'mwhte'] == 1:
        out_flag.loc[index, 'fbmw'] = 1
    elif out_flag.loc[index, 'fblck'] == 1 and out_flag.loc[
        index, 'mblck'] == 1:
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

out_flag.drop(['checking'], axis=1, inplace=True, errors='ignore')
out_flag['lbwght'] = np.log(out_flag['bwght'])

###############################################################################
# Export data to Excel
###############################################################################
out_flag.to_excel(output_folder / 'clean data.xlsx')
