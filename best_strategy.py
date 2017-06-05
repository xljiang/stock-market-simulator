"""
Generate Best Possible Strategy

Created on Tue Apr  4 23:29:43 2017
@author: Xiaolu
"""

import csv

import pandas as pd

from indicators import get_prices


def test_run():
    # Input data
    dates_in_sample = pd.date_range('2008-01-01', '2009-12-31')
    symbols = ['AAPL']
    prices = get_prices(symbols, dates_in_sample)[0]

    order_list = []
    sym = 'AAPL'

    for day, next_day in zip(prices.index[:-1], prices.index[1:]):
        if prices.ix[day, sym] < prices.ix[next_day, sym]:
            order_list.append([day.date(), sym, 'BUY', 200])
            order_list.append([next_day.date(), sym, 'SELL', 200])

        elif prices.ix[day, sym] > prices.ix[next_day, sym]:
            order_list.append([day.date(), sym, 'SELL', 200])
            order_list.append([next_day.date(), sym, 'BUY', 200])

    # write orders to csv file
    with open("orders-bestpossible.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Symbol', 'Order', 'Shares'])
        writer.writerows(order_list)
    f.close()


if __name__ == "__main__":
    test_run()
