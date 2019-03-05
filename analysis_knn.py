import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def run_knn_analysis(x_train, x_test, y_train, y_test):
    knn_reg = KNeighborsRegressor(algorithm='auto', n_neighbors=1)

    # Teaching (fitting) the algorithm based on the training data
    knn_reg.fit(x_train, y_train)

    # Predicting on the X_data that the model has never seen before
    y_pred = knn_reg.predict(x_test)

    # Printing out prediction values for each test observation
    # print(f"Test set predictions: {y_pred}")

    # Calling the score method, which compares the predicted values to the actual values
    y_score = knn_reg.score(x_test, y_test)  # basically an R square value

    # The score is directly comparable to R-Square
    # print(y_score)

    ###############################################################################
    # How Many Neighbors?
    ###############################################################################

    # Loop to choose the best neighbor
    neighbors = np.arange(1, 51)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        # loop though neighbors to find the accuracy rate for each neighbors for
        # both train and test data
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)

        train_accuracy[i] = knn.score(x_train, y_train)
        test_accuracy[i] = knn.score(x_test, y_test)

    # plot the accuracy
    plt.title('k-NN: varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label='Test Accuracy')
    plt.plot(neighbors, train_accuracy, label='Train Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    ###############################################################################
    # KNN regression analysis
    ###############################################################################

    # choose the best neighbor
    neighbors = np.arange(1, 30)
    train_accuracy_reg = np.empty(len(neighbors))
    test_accuracy_reg = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        # loop though neighbors to find the accuracy rate for each neighbors for
        # both train and test data
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(x_train, y_train)

        train_accuracy_reg[i] = knn.score(x_train, y_train)
        test_accuracy_reg[i] = knn.score(x_test, y_test)

    # plot the accuracy
    plt.title('k-NN Regression: varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy_reg, label='Test Accuracy')
    plt.plot(neighbors, train_accuracy_reg, label='Train Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # print(test_accuracy)
    # print(max(test_accuracy))
    #
    # print(np.argmax(test_accuracy))

    # Building a model with k = 5
    knn_reg = KNeighborsRegressor(algorithm='auto',
                                  n_neighbors=10)

    # Fitting the model based on the training data
    knn_reg.fit(x_train, y_train)

    # Scoring the model
    # y_score = knn_reg.score(x_test, y_test)
    # The score is directly comparable to R-Square
    print('Training Score:', knn_reg.score(x_train, y_train).round(4))
    print('Testing Score:', knn_reg.score(x_test, y_test).round(4))
    # 0.667!

    # print(f"""Our base to compare other models is {knn_reg.score(x_test, y_test).round(3)}.
    # This base helps us evaluate more complicated models and lets us consider
    # tradeoffs between accuracy and interpretability.""")
