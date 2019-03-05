# Standard library imports
from pathlib import Path

# Third party imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# declare directory paths
data_folder = Path('data/')
figs_folder = Path('figs/')

file = data_folder / 'clean data.xlsx'
data = pd.read_excel(file)

print(data.columns)

########################
# Visual EDA (Histograms)
########################

# Birth Weight (grams)
sns.distplot(data['bwght'],
             bins=20,
             color='g')

plt.xlabel('Birth Weight')
plt.savefig(figs_folder / 'Birth Weight.png')

# Male
sns.distplot(data['male'],
             bins=20,
             color='g')

plt.xlabel('Male')
plt.savefig(figs_folder / 'Birth Weight.png')

# mage-fage-meduc-feduc
plt.subplot(2, 2, 1)
sns.distplot(data['mage'],
             bins=20,
             color='g')

plt.xlabel('Mother Age')

plt.subplot(2, 2, 2)
sns.distplot(data['fage'],
             bins=20,
             color='g')

plt.xlabel('Father Age')

plt.subplot(2, 2, 3)
sns.distplot(data['meduc'],
             bins=10,
             color='r')

plt.xlabel('Mother Education')

plt.subplot(2, 2, 4)
sns.distplot(data['feduc'],
             bins=10,
             color='r')

plt.xlabel('Father Education')

plt.tight_layout()
plt.savefig(figs_folder / 'mage-fage-meduc-feduc.png')

plt.show()

# monpre-npvis-cigs-drink
plt.subplot(2, 2, 1)
sns.distplot(data['monpre'],
             bins=20,
             color='orange')

plt.xlabel('month prenatal care began')

plt.subplot(2, 2, 2)
sns.distplot(data['npvis'],
             bins=20,
             color='g')

plt.xlabel('total number of prenatal visits')

plt.subplot(2, 2, 3)
sns.distplot(data['cigs'],
             bins=10,
             color='orange')

plt.xlabel('Avg cigarettes per day')

plt.subplot(2, 2, 4)
sns.distplot(data['drink'],
             bins=10,
             color='r')

plt.xlabel('Avg drinks per week')

plt.tight_layout()
plt.savefig(figs_folder / 'monpre-npvis-cigs-drink.png')

plt.show()
