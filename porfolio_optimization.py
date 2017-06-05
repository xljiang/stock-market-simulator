"""
Optimize a stock portfolio by find out how to allocate to each stock to maximize performance, evaluated by Sharpe ratio.
Also give out portfolio analysis.

@author: Xiaolu
"""

import datetime as dt

import numpy as np
import pandas as pd
import scipy.optimize as spo

from util import get_data, plot_data


def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    # allocs = np.asarray([0.2, 0.2, 0.3, 0.3, 0.0])

    # initiallize allocs
    allocs_guess = np.full(len(syms), 1.0 / len(syms))
    # constrains
    cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})  # 1 minus the sum of all variables must be 0
    # bounds
    bnds = tuple((0, 1) for x in allocs_guess)
    # optimization
    allocs = spo.minimize(min_sharpe_ratio, allocs_guess, args=(prices,), \
                          method='SLSQP', bounds=bnds, \
                          constraints=cons)

    # Get optimized daily portfolio value and cr, adr, sddr, sr
    port_val = compute_port_val(prices, allocs.x)
    cr, adr, sddr, sr = compute_portfolio_stats(prices, allocs.x)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        port_val_normed = port_val / port_val.ix[0, :]
        prices_SPY_normed = prices_SPY / prices_SPY.ix[0, :]
        df_temp = pd.concat([port_val_normed, prices_SPY_normed], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily portfolio value and SPY", xlabel="Date", ylabel="Price")
        pass

    return allocs.x, cr, adr, sddr, sr


def compute_port_val(prices, allocs):
    port_normed = prices / prices.ix[0, :]
    port_norm_alloced = port_normed * allocs
    port_val = port_norm_alloced.sum(axis=1)
    return port_val


def compute_portfolio_stats(prices, allocs, rfr=0.0, sf=252.0):
    # port_val: daily portofolio vlaue
    port_val = compute_port_val(prices, allocs)

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


def min_sharpe_ratio(allocs, prices):
    return (-1) * compute_portfolio_stats(prices, allocs)[3]


def test_code():
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    test_code()
