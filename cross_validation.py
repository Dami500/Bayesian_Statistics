from ensemble_prediction import obtain_lagged_series
import numpy as np
import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def validation_set_poly(random_seeds, degrees, X, y):
    """
    Use the train_test_split method to create a training set and a validation set (50% in each)
    using "random_seeds" separate random samplings over linear regression models of varying flexibility
    """
    sample_dict = dict([("seed_%s" % i,[]) for i in range(1, random_seeds+1)])
    # Loop over each random splitting into a train-test split
    for i in range(1, random_seeds+1):
        print("Random: %s" % i)
        # Increase degree of linear regression polynomial order
        for d in range(1, degrees + 1):
            print("Degree: %s" % d)
            # Create the model, split the sets and fit it
            polynomial_features = PolynomialFeatures(degree=d, include_bias=False)
            linear_regression = LinearRegression()
            model = Pipeline([("polynomial_features", polynomial_features),("linear_regression", linear_regression)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
            model.fit(X_train, y_train)
            # Calculate the test MSE and append to the dictionary of all test curves
            y_pred = model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred)
            sample_dict["seed_%s" % i].append(test_mse)
        # Convert these lists into numpy arrays to perform averaging
        sample_dict["seed_%s" % i] = np.array(sample_dict["seed_%s" % i])
    # Create the "average test MSE" series by averaging the test MSE for each degree of the linear regression model,
    # across all random samples
    sample_dict["avg"] = np.zeros(degrees)
    for i in range(1, random_seeds + 1):
        sample_dict["avg"] += sample_dict["seed_%s" % i]
    sample_dict["avg"] /= float(random_seeds)
    return sample_dict

def plot_test_error_curves_vs(sample_dict, random_seeds, degrees):
    fig, ax = plt.subplots()
    ds = range(1, degrees+1)
    for i in range(1, random_seeds+1):
        ax.plot(ds, sample_dict["seed_%s" % i], lw=2, label='Test MSE - Sample %s' % i)
    ax.plot(ds, sample_dict["avg"], linestyle= '--', color="k", lw=3, label= 'Avg Test MSE')
    ax.legend(loc=0)
    ax.set_xlabel('Degree of Polynomial Fit')
    ax.set_ylabel('Mean Squared Error')
    fig.set_facecolor('white')
    plt.show()

if __name__ == '__main__':
    symbol = "AMZN"
    start_date = datetime.datetime(2004, 1, 1)
    end_date = datetime.datetime(2016, 10, 27)
    df = obtain_lagged_series(symbol, start_date, end_date, lags=10)
    X = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10']]
    y = df['direction']
    samples = validation_set_poly(10, 3 , X, y)
    plot_test_error_curves_vs(samples, 10, 3)
