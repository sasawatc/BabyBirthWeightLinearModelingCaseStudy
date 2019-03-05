import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def run_decision_tree_analysis(x_train, x_test, y_train, y_test):
    tree_full_try = DecisionTreeRegressor(random_state=508)
    tree_full_try.fit(x_train, y_train)

    print('Training Score:', tree_full_try.score(x_train, y_train).round(4))
    print('Testing Score:', tree_full_try.score(x_test, y_test).round(4))

    # Creating a tree with only two levels.
    tree_2 = DecisionTreeRegressor(max_depth=3,
                                   random_state=508)

    tree_2_fit = tree_2.fit(x_train, y_train)

    print('Training Score:', tree_2.score(x_train, y_train).round(4))
    print('Testing Score:', tree_2.score(x_test, y_test).round(4))

    def plot_feature_importances(model, train=x_train, export=False):
        fig, ax = plt.subplots(figsize=(12, 9))
        n_features = train.shape[1]
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.yticks(pd.np.arange(n_features), train.columns)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")

        if export:
            plt.savefig(model + '.png')

    plot_feature_importances(tree_full_try,
                             train=x_train,
                             export=False)
