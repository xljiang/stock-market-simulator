"""
Market simulator
Create a market simulator that accepts trading orders and keeps track of a portfolio's value over time,
and then assesses the performance of the portfolio.

@author Xiaolu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indicators import get_prices
from util import get_data


def compute_portvalue(symbols, dates, orders_file, start_val=100000):
    # Read in order.csv data
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    # sort orders by date
    orders = orders.sort_index()

    # Get prices of a given symbols and date range
    prices = get_prices(symbols, dates)[0]

    # Step 1: 'prices' dataframe
    prices['CASH'] = 1.0  # add Cash -- all = 1.0
    # print prices.ix[:10,:]

    # Step 3: 'holdings' dataframe
    # initialize 'holdings' from 'prices', fill columns by 0
    # then add start value to the first colunm of cash
    holdings = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    holdings.ix[0, -1] = start_val  # should be holdings.ix[:,-1] = start_val

    # read orders from the 'orders' dataframe
    # calculate leverage, update holdings, etc.
    # if one order void leverage, then cancel all trades on that day
    for row in orders.itertuples():
        # pass orders in non trading days
        if row.Index not in prices.index:
            continue

        # read in shares
        if row.Order == 'BUY':
            shares = row.Shares
        else:
            shares = -row.Shares

        # calculate leverage_before the order placed
        hld_before = holdings.ix[row.Index, :].copy()
        val_before = prices.ix[row.Index, :] * hld_before
        leverage_before = sum(abs(val_before[:-1])) / sum(val_before)

        # calculate leverage after the order placed
        hld = holdings.ix[row.Index, :].copy()
        hld[row.Symbol] += shares
        hld['CASH'] -= prices.ix[row.Index, row.Symbol] * shares
        val = prices.ix[row.Index, :] * hld
        leverage = sum(abs(val[:-1])) / sum(val)
        # print "lev-bef: ", leverage_before, "lev: ", leverage

        if True:  # leverage < 1.5 or leverage < leverage_before:
            # make order happen
            holdings.ix[row.Index, row.Symbol] += shares
            holdings.ix[row.Index, 'CASH'] -= prices.ix[row.Index, row.Symbol] * shares
            holdings.ix[row.Index:, :] = holdings.ix[row.Index, :].values

    # Step 4: 'values' dataframe
    values = prices * holdings

    # Step 5: 'portvals'
    portvals = values.sum(axis=1)

    return portvals


def compute_portfolio_stats(port_val, rfr=0.0, sf=252.0):
    # cr: Cumulative return
    cr = port_val[-1] / port_val[0] - 1
    # compute daily return
    daily_rets = port_val.copy()
    daily_rets[1:] = (port_val[1:] / port_val[:-1].values) - 1
    daily_rets.ix[0] = 0
    # adr: Average daily return
    adr = daily_rets[1:].mean()
    # sddr: Standard deviation of daily return
    sddr = daily_rets[1:].std()
    # sr: Sharpe Ratio
    sr = np.sqrt(sf) * (daily_rets[1:] - rfr).mean() / sddr
    return cr, adr, sddr, sr


def compute_portvals_SPY(start_date, end_date, start_val):
    dates = pd.date_range(start_date, end_date)
    prices_SPY = get_data(['SPY'], dates)
    portvals_SPY = (prices_SPY / prices_SPY.ix[0, :] * start_val).sum(axis=1)
    print portvals_SPY
    return portvals_SPY


def test_run_insample():
    # Input orders from csv file, symbol, data range and start value
    of_benchmark = "orders-benchmark.csv"
    of_rule = "orders-rule-based_insample.csv"
    of_ml = "orders-ml-based_insample.csv"
    sv = 100000
    dates_in_sample = pd.date_range('2008-01-01', '2009-12-31')
    symbols = ['AAPL']

    # Process orders
    portvals_benchmark = compute_portvalue(symbols, dates_in_sample, of_benchmark, sv)
    portvals_benchmark_normed = portvals_benchmark / portvals_benchmark.ix[0, :]

    portvals_rule = compute_portvalue(symbols, dates_in_sample, of_rule, sv)
    portvals_rule_normed = portvals_rule / portvals_rule.ix[0, :]

    portvals_ml = compute_portvalue(symbols, dates_in_sample, of_ml, sv)
    portvals_ml_normed = portvals_ml / portvals_ml.ix[0, :]

    # Get portfolio stats
    start_date = portvals_benchmark.index[0]
    end_date = portvals_benchmark.index[-1]
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = \
        compute_portfolio_stats(portvals_benchmark)

    cum_ret_rule, avg_daily_ret_rule, std_daily_ret_rule, sharpe_ratio_rule = \
        compute_portfolio_stats(portvals_rule)

    cum_ret_ml, avg_daily_ret_ml, std_daily_ret_ml, sharpe_ratio_ml = \
        compute_portfolio_stats(portvals_ml)

    # Print portfolio stats
    print "Benchmark statistics of in sample data"
    print "Date Range: {} to {}".format(start_date, end_date)
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio_bench)
    print "Cumulative Return of Fund: {}".format(cum_ret_bench)
    print "Standard Deviation of Fund: {}".format(std_daily_ret_bench)
    print "Average Daily Return of Fund: {}".format(avg_daily_ret_bench)
    print "Final Portfolio Value: {}".format(portvals_benchmark[-1])
    print "------------"
    print "Rule based statistics of in sample data"
    print "Date Range: {} to {}".format(start_date, end_date)
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio_rule)
    print "Cumulative Return of Fund: {}".format(cum_ret_rule)
    print "Standard Deviation of Fund: {}".format(std_daily_ret_rule)
    print "Average Daily Return of Fund: {}".format(avg_daily_ret_rule)
    print "Final Portfolio Value: {}".format(portvals_rule[-1])
    print "------------"
    print "ML based statistics of in sample data"
    print "Date Range: {} to {}".format(start_date, end_date)
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio_ml)
    print "Cumulative Return of Fund: {}".format(cum_ret_ml)
    print "Standard Deviation of Fund: {}".format(std_daily_ret_ml)
    print "Average Daily Return of Fund: {}".format(avg_daily_ret_ml)
    print "Final Portfolio Value: {}".format(portvals_ml[-1])
    print "------------"

    # Plotting
    plt.plot(portvals_ml_normed, 'g', lw=1, label='ML Based')
    plt.plot(portvals_rule_normed, 'b', lw=1, label='Rule Based')
    plt.plot(portvals_benchmark_normed, 'k', lw=1, label='Benchmark')
    plt.title('ML Based Strategy: In sample portfolio')
    plt.ylabel('Normalized Portfolio')
    plt.legend(loc=2, prop={'size': 9})
    # plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    # plot ML entries
    orders = pd.read_csv(of_ml, index_col='Date', parse_dates=True, na_values=['nan'])
    entries = orders.iloc[::2]  # only even
    for row in entries.itertuples():
        if row.Order == 'BUY':
            plt.axvline(x=row.Index, color='green', lw=1)
        else:
            plt.axvline(x=row.Index, color='red', lw=1)
    plt.savefig('strategy-ml-in-sample.png', dpi=150, format='png')


if __name__ == "__main__":
    test_run_insample()
