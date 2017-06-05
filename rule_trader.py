"""
Rule Based Trader based on historical technical indicators data.
Output: orders csv files

Created on Tue Apr  4 23:29:43 2017
@author: Xiaolu
"""

import csv

import numpy as np
import pandas as pd

from indicators import get_prices, get_all_indicators


def build_orders(symbols, dates, holding_days, window, output_filename):
    # Get prices (adj close and tipical prices)
    prices = get_prices(symbols, dates)[0]
    tp_prices = get_prices(symbols, dates)[2]

    # get all indicator values (not normalized)
    indicators = get_all_indicators(prices, tp_prices, window)[0]
    # bollinger band precent
    bbp = indicators['BBP']
    # price/sma
    price_sma_ratio = indicators['PRICE_SMA']
    # momentum
    momentum = indicators['MOMENTUM']
    # cci
    cci = indicators['CCI']

    ### Use the four indicators to make rule based decision

    # Holdings starts as a NaN array of the same shape/index as price.
    orders = prices.copy()
    orders.ix[:, :] = np.NaN

    # Apply our entry order conditions all at once.  This represents our TARGET SHARES
    # at this moment in time, not actual orders.

    # orders didn't consider holding_days at this stage
    orders.ix[0, :] = 200  # benchmark entry
    orders[(momentum > 0) & ((price_sma_ratio < 0.95) | (bbp < 0) | (cci < -50))] = 200
    orders[(momentum < 0) & ((price_sma_ratio > 1.05) | (bbp > 1) | (cci > 150))] = -200
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)

    # construct order_list from orders dataframe, also consider holding_days
    order_list = []  # a list of both entry and exit orders

    count = holding_days
    for sym in symbols:
        for i in range(orders.shape[0]):
            if count > 0 and count < holding_days:
                count -= 1
                continue
            if orders.ix[i, sym] == 0:
                continue
            if orders.ix[i, sym] != 0:
                count = holding_days - 1
                exit_i = i + holding_days
                # write into order_list
                if orders.ix[i, sym] > 0:
                    order_list.append([orders.index[i].date(), sym, 'BUY', 200])
                    if exit_i < orders.shape[0]:
                        order_list.append([orders.index[exit_i].date(), sym, 'SELL', 200])
                elif orders.ix[i, sym] < 0:
                    order_list.append([orders.index[i].date(), sym, 'SELL', 200])
                    if exit_i < orders.shape[0]:
                        order_list.append([orders.index[exit_i].date(), sym, 'BUY', 200])

    # print order_list        
    for order in order_list:
        print "	".join(str(x) for x in order)

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

    # in sample
    dates_in_sample = pd.date_range('2008-01-01', '2009-12-31')
    output_filename = "orders-rule-based_insample.csv"

    print "Constructing Price/SMA, BB%, CCI, Momentum for in sample period"
    build_orders(symbols, dates_in_sample, holding_days, window, output_filename)

    # out sample
    dates_out_sample = pd.date_range('2010-01-01', '2011-12-31')
    output_filename = "orders-rule-based_outsample.csv"

    print "Constructing Price/SMA, BB%, CCI, Momentumfor out sample period"
    build_orders(symbols, dates_out_sample, holding_days, window, output_filename)
