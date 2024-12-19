import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
import mysql.connector as msc
import warnings

warnings.filterwarnings('ignore')
db_host = 'localhost'
db_user = 'sec_user'
db_pass = 'Damilare20$'
db_name = 'securities_master'
plug ='caching_sha2_password'
con = msc.connect(host=db_host, user=db_user, password=db_pass, db=db_name, auth_plugin= plug)

def obtain_lagged_series(symbol, start_date, end_date, lags = 5):
    """
    This creates a Pandas DataFrame that stores the percentage returns of the adjusted closing value of
    a stock obtained from Yahoo Finance, along with a number of lagged returns from the prior trading days
    (lags defaults to 5 days). Trading volume, as well as the Direction from the previous day, are also included
    """
    symbol_select_str = """SELECT securities_master.symbol.id
                    FROM securities_master.symbol
                    where securities_master.symbol.ticker = '%s' """ % symbol
    df = pd.read_sql_query(symbol_select_str, con)
    symbol_id = df.iloc[0, 0]
    f_start_date = start_date.strftime('%Y-%m-%d')
    f_end_date = end_date.strftime('%Y-%m-%d')
    price_select_str = """SELECT distinct securities_master.daily_price.close_price,
                          securities_master.daily_price.volume
                          FROM securities_master.daily_price
                          where securities_master.daily_price.symbol_id = '%d' and
                          securities_master.daily_price.price_date >= '%s' and 
                          securities_master.daily_price.price_date <= '%s'
                          """ % (symbol_id, f_start_date, f_end_date)
    df = pd.read_sql_query(price_select_str, con)
    # Create the new lagged DataFrame
    df_lag = pd.DataFrame(index=df.index)
    df_lag['today'] = df['close_price']
    df_lag['volume'] = df['volume']
    # Create the shifted lag series of prior trading period close values
    for i in range(lags):
        df_lag['lag%s' % str(i+1)] = df_lag['today'].shift(i+1)
    # Create the returns DataFrame
    df_returns = pd.DataFrame(index=df_lag.index)
    df_returns['volume'] = df_lag['volume']
    df_returns['percent_change'] = df_lag['today'].pct_change()*100
    df_returns['percent_change'] = df_returns['percent_change'].fillna(0)
    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in Scikit-Learn)
    for i, x in enumerate(df_returns["percent_change"]):
        if abs(x) < 0.0001:
            df_returns['percent_change'][i] = 0.0001
    # Create the lagged percentage returns columns
    for i in range(lags):
        df_returns["lag%s" % str(i + 1)] = df_lag["lag%s" % str(i + 1)].pct_change() * 100.0
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    df_returns["direction"] = np.sign(df_returns["percent_change"])
    return df_returns.fillna(0)

if __name__ == '__main__':
    # Set the random seed, number of estimators
    # and the "step factor" used to plot the graph of MSE
    # for each method
    random_state = 42
    n_jobs = 1  # Parallelization factor for bagging, random forests
    n_estimators = 1000
    step_factor = 10
    axis_step = int(n_estimators / step_factor)
    # Download ten years worth of Amazon
    # adjusted closing prices
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    amzn = obtain_lagged_series("AMZN", start, end, lags=3)
    # Use the first three daily lags of AMZN closing prices and scale the data to lie within -1 and +1 for comparison
    X = amzn[["lag1", "lag2", "lag3"]]
    y = amzn["direction"]
    X = scale(X)
    y = scale(y)
    # Use the training-testing split with 70% of data in the
    # training data with the remaining 30% of data in the testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    # Pre-create the arrays which will contain the MSE for each particular ensemble method
    estimators = np.zeros(axis_step)
    bagging_mse = np.zeros(axis_step)
    rf_mse = np.zeros(axis_step)
    boosting_mse = np.zeros(axis_step)
    # Estimate the Bagging MSE over the full number of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Bagging Estimator: %d of %d..." % (step_factor * (i + 1), n_estimators))
        bagging = BaggingRegressor(DecisionTreeRegressor(), n_estimators=step_factor * (i + 1), n_jobs=n_jobs,
        random_state=random_state)
        bagging.fit(X_train, y_train)
        mse = mean_squared_error(y_test, bagging.predict(X_test))
        estimators[i] = step_factor * (i + 1)
        bagging_mse[i] = mse
    # Estimate the Random Forest MSE over the full number of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Random Forest Estimator: %d of %d..." % (step_factor * (i + 1), n_estimators))
        rf = RandomForestRegressor(n_estimators=step_factor * (i + 1), n_jobs=n_jobs, random_state=random_state)
        rf.fit(X_train, y_train)
        mse = mean_squared_error(y_test, rf.predict(X_test))
        estimators[i] = step_factor * (i + 1)
        rf_mse[i] = mse
    # Estimate the AdaBoost MSE over the full number of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Boosting Estimator: %d of %d..." % (step_factor * (i + 1), n_estimators))
        boosting = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=step_factor * (i + 1),
                                     random_state=random_state,learning_rate=0.01)
        boosting.fit(X_train, y_train)
        mse = mean_squared_error(y_test, boosting.predict(X_test))
        estimators[i] = step_factor * (i + 1)
        boosting_mse[i] = mse

    # Plot the chart of MSE versus number of estimators
    plt.figure(figsize=(8, 8))
    plt.title('Comparison of Bagging, RandomForest, and Boosting', fontsize=16)
    plt.plot(estimators, bagging_mse, linestyle='-', color='black', label='Bagging')
    plt.plot(estimators, rf_mse, linestyle='-', color='blue', label='RandomForest')
    plt.plot(estimators, boosting_mse, linestyle='-', color='red', label='AdaBoost')
    plt.xlabel('Number of Estimators', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    # Legend
    plt.legend(loc='upper right', fontsize=12)
    # Display the plot
    plt.show()
    plt.show()


