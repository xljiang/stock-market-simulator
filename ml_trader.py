"""
Machine Learning algorithm Based Trader based on historical technical indicators data.
Output: orders csv files

Created on Tue Apr  4 23:29:43 2017
@author: Xiaolu
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bootstrap_aggr as bl
import random_tree_model as rt
from indicators import get_prices, get_all_indicators


def gen_data(symbols, dates, holding_days, window, Ybuy, Ysell):
    """
    generate normalized indicator values (dataX), 
    and the classification based on holding_days return (dataY)
    """
    # Get prices (adj close and tipical prices)
    prices = get_prices(symbols, dates)[0]  # adj close prices
    tp_prices = get_prices(symbols, dates)[2]  # tipical prices

    # get all normalized indicator values as dataX
    dataX = get_all_indicators(prices, tp_prices, window)[1]

    # ret: holding_days return (e.g. 21 days return)
    ret = prices.copy()
    ret[holding_days:] = (prices[holding_days:] / prices[:-holding_days].values) - 1
    ret.ix[:holding_days] = 0

    ### Classification
    # Generate dataY to classification values LONG(1), SHORT(-1) or HOLD(0)
    # based on holding_days return.
    dataY = ret.copy()
    dataY.ix[:, :] = np.NaN
    dataY[(ret > Ybuy)] = 1  # if ret > Ybuy, Long
    dataY[(ret <= Ybuy) & (ret < Ysell)] = -1  # else if ret < Ysell, Short
    dataY.fillna(0, inplace=True)  # else hold, do nothing

    # print ret
    # print dataY

    # convert data from dataframe into np array (to fit for RTLearner I created)
    dataX_m = dataX.as_matrix()  # convert to np array
    dataY_m = dataY.as_matrix().T[0]  # convert to 1d np array

    return dataX_m, dataY_m


def gen_decision_tree_learner(leaf_size, dataX, dataY):
    """
    generate a decision tree learner
    """
    learner = rt.RTLearner(leaf_size, verbose=False)
    learner.addEvidence(dataX, dataY)

    return learner


def gen_bag_learner(bag_size, leaf_size, dataX, dataY):
    """
    generate a bag learner
    """
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, \
                            bags=bag_size, boost=False, verbose=False)
    learner.addEvidence(dataX, dataY)
    return learner


def build_order_list(dataY, symbols, dates, holding_days, output_filename):
    """
    construct order_list from dataY, also consider holding_days
    dataY: 1d np array, Y data (classification) for each day with Long, short or do nothing result
    """
    # get prices for index template
    prices = get_prices(symbols, dates)[0]

    dataY[0] = 1  # benchmark entry

    # build order_list     
    order_list = []  # a list of both entry and exit orders

    count = holding_days
    for sym in symbols:
        for i in range(dataY.shape[0]):
            if count > 0 and count < holding_days:
                count -= 1
                continue
            if dataY[i] == 0:
                continue
            if dataY[i] != 0:
                count = holding_days - 1
                exit_i = i + holding_days
                # write into order_list
                if dataY[i] > 0:
                    order_list.append([prices.index[i].date(), sym, 'BUY', 200])
                    if exit_i < dataY.shape[0]:
                        order_list.append([prices.index[exit_i].date(), sym, 'SELL', 200])
                elif dataY[i] < 0:
                    order_list.append([prices.index[i].date(), sym, 'SELL', 200])
                    if exit_i < dataY.shape[0]:
                        order_list.append([prices.index[exit_i].date(), sym, 'BUY', 200])
    """
    # print order_list        
    for order in order_list:
        print "	".join(str(x) for x in order)
    """

    # write order_list to csv file
    with open(output_filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Symbol', 'Order', 'Shares'])
        writer.writerows(order_list)
    f.close()


if __name__ == "__main__":
    symbols = ['AAPL']
    holding_days = 21
    window = 20

    # parameters
    leaf_size = 5
    bag_size = 50
    Ybuy = 0.0
    Ysell = -0.14

    # in sample
    dates_in_sample = pd.date_range('2008-01-01', '2009-12-31')
    output_filename = "orders-ml-based_insample.csv"

    print "ML - In sample period"
    trainX, trainY = gen_data(symbols, dates_in_sample, holding_days, window, Ybuy, Ysell)
    # learner = gen_decision_tree_learner(leaf_size, trainX, trainY)
    learner = gen_bag_learner(bag_size, leaf_size, trainX, trainY)
    pred_trainY = learner.query(trainX)
    build_order_list(pred_trainY, symbols, dates_in_sample, holding_days, output_filename)



    # out sample
    dates_out_sample = pd.date_range('2010-01-01', '2011-12-31')
    output_filename = "orders-ml-based_outsample.csv"

    print "ML - Out sample period"
    testX, testY = gen_data(symbols, dates_out_sample, holding_days, window, Ybuy, Ysell)
    pred_testY = learner.query(testX)
    build_order_list(pred_testY, symbols, dates_out_sample, holding_days, output_filename)

    ### ------- visualization part 5-2, 5-3 ----------

    bbp = trainX[:, 0]
    cci = trainX[:, 3]

    # plot for part 5-2
    df_train = pd.DataFrame(dict(bbp=bbp, cci=cci, color=trainY))
    colors = {1: 'green', -1: 'red', 0: 'black'}
    fig, ax = plt.subplots()
    ax.scatter(df_train['bbp'], df_train['cci'], c=df_train['color'].apply(lambda x: colors[x]))
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel("BBP")
    ax.set_ylabel("CCI")
    ax.set_title("Training Data for ML strategy")
    plt.savefig('scatter-ml-based-train.png', dpi=150, format='png')
    plt.show()

    # plot for part 5-3
    # print pred_trainY
    df_pred = pd.DataFrame(dict(bbp=bbp, cci=cci, color=pred_trainY))
    colors = {1: 'green', -1: 'red', 0: 'black'}
    fig, ax = plt.subplots()
    ax.scatter(df_pred['bbp'], df_pred['cci'], c=df_pred['color'].apply(lambda x: colors[x]))
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel("BBP")
    ax.set_ylabel("CCI")
    ax.set_title("Predicted Data for ML strategy")
    plt.savefig('scatter-ml-based-pred.png', dpi=150, format='png')
    plt.show()
