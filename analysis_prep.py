from sklearn.model_selection import train_test_split


def prepare_analysis(df):
    for miss_col in df:
        fill = df[miss_col].median()
        df[miss_col] = df[miss_col].fillna(fill)


def check_correlations(df):
    df_corr = df.corr().round(2)
    df_corr.loc['bwght'].sort_values(ascending=False)
    print(df_corr)


def flag_outlier(df, col_name, outlier_val, outlier_prefix='lot_'):
    outlier_col_name = outlier_prefix + col_name
    df[outlier_col_name] = 0
    for val in enumerate(df.loc[:, col_name]):
        if val[1] >= outlier_val:
            df.loc[val[0], outlier_col_name] = 1
    return df


def flag_outliers(df):
    # Mother's Age
    df_new = flag_outlier(df, col_name='mage', outlier_val=35)

    # Father's Age
    df_new = flag_outlier(df_new, col_name='fage', outlier_val=29)

    # AVG Cigarettes per Day
    df_new = flag_outlier(df_new, col_name='cigs', outlier_val=1)

    # Father's Education (years)
    df_new = flag_outlier(df_new, col_name='feduc', outlier_val=12)

    return df_new


def get_xy_test_train_split(df, test_size, random_state, independent_var_lst):
    target = df.loc[:, 'bwght']

    independent_vars = df.loc[:, independent_var_lst]

    return train_test_split(
        independent_vars,
        target,
        test_size=test_size,
        random_state=random_state)
