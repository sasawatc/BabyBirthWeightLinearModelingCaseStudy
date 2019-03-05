from sklearn import linear_model


def run_lr_analysis(x_train, x_test, y_train, y_test):
    lr = linear_model.LinearRegression(fit_intercept=True,
                                       normalize=False,
                                       copy_X=True,
                                       n_jobs=None)
    lr_fit = lr.fit(x_train, y_train)
    lr_pred = lr_fit.predict(x_test)
    y_score_ols_optimal = lr_fit.score(x_test, y_test)

    print('Training Score:', lr.score(x_train, y_train).round(4))
    print('Testing Score:', lr.score(x_test, y_test).round(4))
