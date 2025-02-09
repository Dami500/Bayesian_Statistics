from retrieve_data import get_prices_id, get_prices
import pandas as pd
import mysql.connector as msc
import warnings
import copy
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.dates import (DateFormatter, WeekdayLocator, DayLocator, MO)
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
db_host = 'localhost'
db_user = 'sec_user'
db_pass = 'Damilare20$'
db_name = 'securities_master'
plug ='caching_sha2_password'
con = msc.connect(host=db_host, user=db_user, password=db_pass, db=db_name, auth_plugin= plug)

def get_open_normalised_prices(symbol, start, end):
    """
    Obtains a pandas DataFrame containing open normalised prices
    for high, low and close for a particular equities symbol
    from Yahoo Finance. That is, it creates High/Open, Low/Open
    and Close/Open columns.
    """
    location = get_prices_id(symbol)
    prices = get_prices(location)[0]
    df = prices[(prices['price_date'] >= start) & (prices['price_date'] <= end)]
    df["H/O"] = df["high_price"]/df["open_price"]
    df["L/O"] = df["low_price"]/df["open_price"]
    df["C/O"] = df["close_price"]/df["open_price"]
    df.drop(["open_price", "high_price", "low_price","close_price", "volume"],axis=1, inplace=True)
    return df

def plot_candlesticks(data, since, end):
    """
    Plot a candlestick chart of the prices,
    appropriately formatted for dates
    """
    # Copy and reset the index of the dataframe to only use a subset of the data for plotting
    df = copy.deepcopy(data)
    df = df.loc[(df['price_date'] >= since) &(df['price_date'] <= end)]
    df.reset_index(inplace=True)
    df['date_fmt'] = df['price_date'].apply(lambda date: mdates.date2num(date.to_pydatetime()))
    # Set the axis formatting correctly for dates
    mondays = WeekdayLocator(byweekday=0)
    alldays = DayLocator()
    weekFormatter = DateFormatter('%b %d')
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    # Plot the candlestick OHLC chart using black for up days and red for down days
    csticks = candlestick_ohlc(ax,df[['date_fmt', 'open_price', 'high_price', 'low_price', 'close_price']].values, width=0.6, colorup='#000000',
        colordown='#ff0000')
    ax.set_facecolor((1, 1, 0.9))  # Updated method for setting axis background color
    ax.xaxis_date()
    plt.setp(
        plt.gca().get_xticklabels(),
        rotation=45,
        horizontalalignment='right')
    plt.show()

def plot_3d_normalised_candles(data, labels):
    """
    Plot a 3D scatter chart of the open-normalised bars
    highlighting the separate clusters by colour
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Ensure labels are numeric
    labels = labels.astype(float)
    # Scatter plot in 3D
    ax.scatter(data["H/O"], data["L/O"], data["C/O"],
               c=labels)
    ax.set_xlabel('High/Open')
    ax.set_ylabel('Low/Open')
    ax.set_zlabel('Close/Open')
    plt.show()



def plot_cluster_ordered_candles(data):
    """
    Plot a candlestick chart ordered by cluster membership
    with the dotted blue line representing each cluster
    boundary.
    """
    mondays = WeekdayLocator(byweekday=0)  # Define Monday as major tick
    alldays = DayLocator()  # All days as minor tick
    weekFormatter = DateFormatter("")  # Blank for custom formatting

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    # Sort the data by cluster membership
    df = copy.deepcopy(data)
    df.sort_values(by="Cluster", inplace=True)
    df.reset_index(inplace=True)

    # Create a new column for cluster indices and boundaries
    df["clust_index"] = df.index
    df["clust_change"] = df["Cluster"].diff()
    change_indices = df[df["clust_change"] != 0]

    # Plot the candlestick chart
    csticks = candlestick_ohlc(
        ax, df[["clust_index", 'open_price', 'high_price', 'low_price', 'close_price']].values, width=0.6,
        colorup='#000000', colordown='#ff0000')
    ax.set_facecolor((1, 1, 0.9))  # Light background color
    # Add cluster boundaries as blue dotted lines
    for row in change_indices.iterrows():
        plt.axvline(
            row[1]["clust_index"], linestyle="dashed", c="blue"
        )

    plt.xlim(0, len(df))
    plt.setp(
        plt.gca().get_xticklabels(),
        rotation=45, horizontalalignment='right'
    )
    plt.show()

def create_follow_cluster_matrix(data):
    """
    Creates a k x k matrix, where k is the number of clusters
    that shows when cluster j follows cluster i.
    """
    # Shift the cluster column to get the next day's cluster
    data["ClusterTomorrow"] = data["Cluster"].shift(-1)
    data.dropna(inplace=True)  # Remove rows where 'ClusterTomorrow' is NaN
    data["ClusterTomorrow"] = data["ClusterTomorrow"].apply(int)
    # Create combined cluster pairs
    data["ClusterMatrix"] = list(zip(data["Cluster"], data["ClusterTomorrow"]))
    # Count occurrences of each (current_cluster, next_cluster) pair
    cmvc = data["ClusterMatrix"].value_counts()
    # Initialize an empty matrix for k clusters
    k = len(data["Cluster"].unique())
    clust_mat = np.zeros((k, k))
    # Populate matrix with percentages
    for row in cmvc.items():
        clust_mat[row[0]] = row[1] * 100.0 / len(data)
    print("Cluster Follow-on Matrix:")
    print(clust_mat)

if __name__ == "__main__":
    # Obtain S&P500 pricing data from Yahoo Finance
    start = datetime.datetime(2013, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    location = get_prices_id(['SPY'])
    spy = get_prices(location)[0]
    spy = spy.loc[(spy['price_date'] >= start) & (spy['price_date']<= end)]
    # print(spy)
    # Plot last year of price "candles"
    plot_candlesticks(spy, datetime.datetime(2015, 1, 1), datetime.datetime(2015, 12,31))
    # Carry out K-Means clustering with five clusters on the
    # three-dimensional data H/O, L/O and C/O
    sp500_norm = get_open_normalised_prices(['SPY'], start, end)
    # print(sp500_norm)
    # print(len(sp500_norm))
    k = 5
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(sp500_norm[['H/O','L/O','C/O']])
    labels = km.labels_
    # print(labels)
    # print(len(labels))
    spy["Cluster"] = labels
    # print(spy)
    # Plot the 3D normalised candles using H/O, L/O, C/O
    plot_3d_normalised_candles(sp500_norm, labels)
    # Plot the full OHLC candles re-ordered
    # into their respective clusters
    plot_cluster_ordered_candles(spy)
    # Create and output the cluster follow-on matrix
    create_follow_cluster_matrix(spy)
