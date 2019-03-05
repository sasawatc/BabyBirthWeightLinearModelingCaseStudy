# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd

# Local application imports
import analysis_decision_tree
import analysis_knn
import analysis_lr
import analysis_prep

data_folder = Path('data/')

df = pd.read_excel(data_folder / 'clean data.xlsx')
df = analysis_prep.flag_outliers(df)
x_train, x_test, y_train, y_test = analysis_prep.get_xy_test_train_split(df,
                                                                         test_size=0.10,
                                                                         random_state=508,
                                                                         independent_var_lst=['drink',
                                                                                              'cigs',
                                                                                              'mage',
                                                                                              'fage',
                                                                                              'lot_mage',
                                                                                              'lot_fage',
                                                                                              'lot_feduc',
                                                                                              'lot_cigs'])
# Decision Tree
print("-----------------------------------------------------------------")
print("Running Decision Tree:")
analysis_decision_tree.run_decision_tree_analysis(x_train, x_test, y_train, y_test)
print()

# KNN
print("-----------------------------------------------------------------")
print("Running K-Nearest Neighbor:")
analysis_knn.run_knn_analysis(x_train, x_test, y_train, y_test)
print()

# Linear Regression
print("-----------------------------------------------------------------")
print("Running Linear Regression:")
analysis_lr.run_lr_analysis(x_train, x_test, y_train, y_test)
print()
