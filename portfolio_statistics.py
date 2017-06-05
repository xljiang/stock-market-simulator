"""
Compute and evaluate stock portfolios.
Get key statistics: Sharpe ratio, Volatility, Average Daily Return and Cumulative Return.

@author: Xiaolu
"""

import datetime as dt

import numpy as np
import pandas as pd

from util import get_data, plot_data


# Compute and evaluate stock portfolios
def evaluate_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], \
                       allocs=[0.1, 0.2, 0.3, 0.4], \
                       sv=1000000, rfr=0.0, sf=252.0, \
                       gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = compute_port_val(prices, allocs, sv)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val, allocs, rfr, sf)
    print prices
    print prices[-1]
    print prices[0]

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        port_val_normed = port_val / port_val.ix[0, :]
        prices_SPY_normed = prices_SPY / prices_SPY.ix[0, :]
        df_temp = pd.concat([port_val_normed, prices_SPY_normed], \
                            keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily portfolio value and SPY", \
                  xlabel="Date", ylabel="Normalized price")
        # pass

    # Compute end value
    ev = port_val[-1]

    return cr, adr, sddr, sr, ev


def compute_port_val(prices, allocs, sv):
    port_normed = prices / prices.ix[0, :]
    port_norm_alloced = port_normed * allocs
    position_val = port_norm_alloced * sv
    port_val = position_val.sum(axis=1)
    return port_val


def compute_portfolio_stats(prices, allocs, rfr, sf):
    # cr: Cumulative return
    cr = prices[-1] / prices[0] - 1

    # compute daily return
    daily_rets = prices.copy()
    daily_rets[1:] = (prices[1:] / prices[:-1].values) - 1
    daily_rets.ix[0] = 0

    # adr: Average daily return
    adr = daily_rets[1:].mean()

    # sddr: Standard deviation of daily return
    sddr = daily_rets[1:].std()

    # sr: Sharpe Ratio
    sr = np.sqrt(sf) * (daily_rets[1:] - rfr).mean() / sddr

    return cr, adr, sddr, sr


def test_code():
    # TEST ---------------------------------------
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = evaluate_portfolio(sd=start_date, ed=end_date, \
                                               syms=symbols, \
                                               allocs=allocations, \
                                               sv=start_val, \
                                               gen_plot=True)

    # Print statistics
    print "TEST 1---------------------"
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
