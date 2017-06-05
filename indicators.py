"""
Technical Indicators

Indicators and Output of Indicator Graphs
- Bollinger Bands
- Price/SMA
- Momentum
- CCI
    
Created on Mon Apr  3 15:21:35 2017
@author: xiaolu
"""

import matplotlib.pyplot as plt
import pandas as pd

from util import get_data

"""INDICATORS"""


def get_bollinger_bands(values, window):
    rm = get_rolling_mean(values, window)
    rstd = get_rolling_std(values, window)
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    bbp = (values - lower_band) / (upper_band - lower_band)
    return bbp, upper_band, lower_band


def get_price_sma_ratio(values, window):
    rm = get_rolling_mean(values, window)
    return values / rm


def get_momentum(values, n):
    return values / values.shift(n) - 1


def get_cci(tp_values, window):
    rm = get_rolling_mean(tp_values, window)
    rstd = get_rolling_std(tp_values, window)
    # mad = tp_values.mad() # mean absolute deviation
    return (tp_values - rm) / (0.015 * rstd)


"""HELPER FUNCTIOMS"""


def get_rolling_mean(values, window):
    """Resurn rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window=window)


def normalize(values):
    return (values - values.mean()) / values.std()


"""GET DATAFRAMES"""


def get_prices(symbol, dates):
    """return adj colse, normalized adj price, and typical price, normalized typical price"""
    # get Adj close prices, and normalized prices
    prices = get_data(symbol, dates)  # only have trading days use SPY as ref
    prices = prices.ix[:, 1:]  # remove SPY
    prices_normed = prices / prices.ix[0, :]

    # get tipical prices (high + low + close)/3
    prices_high = get_data(symbol, dates, colname='High')
    prices_low = get_data(symbol, dates, colname='Low')
    prices_close = get_data(symbol, dates, colname='Close')

    tp_prices = (prices_high.ix[:, 1:] + prices_low.ix[:, 1:] + prices_close.ix[:, 1:]) / 3
    tp_prices_normed = tp_prices / tp_prices.ix[0, :]

    return prices, prices_normed, tp_prices, tp_prices_normed


def get_all_indicators(prices, tp_prices, window):
    """return indicators and normalized indicators dataframe"""
    # get indicator values
    # sma, tipical sma
    sma = get_rolling_mean(prices, window)
    tp_sma = get_rolling_mean(tp_prices, window)

    # bollinger band
    bbp, upper_band, lower_band = get_bollinger_bands(prices, window)

    # price/sma
    price_sma_ratio = get_price_sma_ratio(prices, window)

    # momentum, n=5
    momentum = get_momentum(prices, 5)
    # cci
    cci = get_cci(tp_prices, window)
    # put all indicators into one df
    indicators = pd.concat([sma, tp_sma, bbp, upper_band, lower_band, price_sma_ratio, \
                            momentum, cci], keys=['SMA', 'TP_SMA', 'BBP', 'UPPER_BB', 'LOWER_BB', \
                                                  'PRICE_SMA', 'MOMENTUM', 'CCI'], axis=1)

    # get normalized indicator values
    bbp_norm = normalize(bbp)
    price_sma_ratio_norm = normalize(price_sma_ratio)
    momentum_norm = normalize(momentum)
    cci_norm = normalize(cci)
    normed_indicators = pd.concat([bbp_norm, price_sma_ratio_norm, momentum_norm, \
                                   cci_norm], keys=['BBP', 'PRICE_SMA', 'MOMENTUM', 'CCI'], axis=1)

    return indicators, normed_indicators


def test_run():
    # Get in sample AAPL prices
    dates = pd.date_range('2008-01-01', '2009-12-31')
    symbol = ['AAPL']
    prices_aapl, prices_aapl_normed, tp_prices_aapl, tp_prices_aapl_normed = \
        get_prices(symbol, dates)

    # get all indicator values
    indicators = get_all_indicators(prices_aapl, tp_prices_aapl, 20)[0]

    sma = indicators['SMA']
    tp_sma = indicators['TP_SMA']
    # bollinger band
    bbp = indicators['BBP']
    upper_band = indicators['UPPER_BB']
    lower_band = indicators['LOWER_BB']
    # price/sma
    price_sma_ratio = indicators['PRICE_SMA']
    # momentum
    momentum = indicators['MOMENTUM']
    # cci
    cci = indicators['CCI']

    # plotting
    # plot price/sma
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylim([60, 300])
    plt.plot(prices_aapl, 'b', lw=1, label='Adj Close')
    plt.plot(sma, 'r', lw=1, label='20-day SMA')
    plt.title('Price/SMA')
    plt.ylabel('Prices')
    plt.legend(loc=2, prop={'size': 9})
    plt.grid(True)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.set_ylim([0.7, 1.3])
    plt.plot(price_sma_ratio, 'k', lw=0.75, linestyle='-', label='Price/SMA')
    plt.legend(loc=2, prop={'size': 9})
    plt.ylabel('Price/SMA')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)

    fig.subplots_adjust(hspace=0)
    plt.savefig('indicator_price-sma.png', dpi=150, format='png')

    # plot bollinger bands and bbp indicator
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylim([60, 300])
    plt.plot(prices_aapl, 'b', lw=1, label='Adj Close')
    plt.plot(sma, 'r', lw=1, label='20-day SMA')
    plt.plot(upper_band, 'g', lw=1, label='upper band')
    plt.plot(lower_band, 'c', lw=1, label='lower band')
    plt.title('Bollinger Bands')
    plt.ylabel('Prices')
    plt.legend(loc=2, prop={'size': 9})
    plt.grid(True)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.set_ylim([-0.2, 1.5])
    plt.plot(bbp, 'k', lw=0.75, linestyle='-', label='b%')
    plt.legend(loc=2, prop={'size': 9})
    plt.ylabel('b%')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)

    fig.subplots_adjust(hspace=0)
    plt.savefig('indicator_bb.png', dpi=150, format='png')

    # plot momentum
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylim([60, 300])
    plt.plot(prices_aapl, 'b', lw=1, label='Adj Close')
    plt.plot(sma, 'r', lw=1, label='20-day SMA')
    plt.title('Momentum')
    plt.ylabel('Prices')
    plt.legend(loc=2, prop={'size': 9})
    plt.grid(True)

    ax2 = fig.add_subplot(212, sharex=ax1)
    # ax2.set_ylim([-0.5,0.5])
    plt.plot(momentum, 'k', lw=0.75, linestyle='-', label='Momentum')
    plt.legend(loc=2, prop={'size': 9})
    plt.ylabel('Momentum')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)

    fig.subplots_adjust(hspace=0)
    plt.savefig('indicator_momentum.png', dpi=150, format='png')

    # plot cci
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylim([60, 300])
    plt.plot(prices_aapl, 'b', lw=1, label='Typical Prices (average of High, Low and Close)')
    plt.plot(tp_sma, 'r', lw=1, label='20-day SMA')
    plt.title('Commodity Channel Index')
    plt.ylabel('Typical Prices')
    plt.legend(loc=2, prop={'size': 9})
    plt.grid(True)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.set_ylim([-200, 220])
    plt.plot(cci, 'k', lw=0.75, linestyle='-', label='CCI')
    plt.legend(loc=2, prop={'size': 9})
    plt.ylabel('CCI')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)

    fig.subplots_adjust(hspace=0)
    plt.savefig('indicator_cci.png', dpi=150, format='png')

    '''
    # plot 4 indicators in the same graph
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_subplot(211)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylim([60,300])
    plt.plot(prices_aapl,'b',lw=1, label='Adj Close')
    plt.plot(sma,'r',lw=1, label='20-day SMA')
    plt.title('4 Indicators')
    plt.ylabel('Prices')
    plt.legend(loc=2, prop={'size':9})
    plt.grid(True)
    
    ax2 = fig.add_subplot(212, sharex=ax1)
    #ax2.set_ylim([-0.2,1.5])
    plt.plot(normalize(bbp),'b',lw=0.75,linestyle='-',label='b%')
    plt.plot(normalize(price_sma_ratio),'k',lw=0.75,linestyle='-',label='Price/SMA')
    cci_norm = normalize(cci)
    plt.plot(cci_norm,'c',lw=0.75,linestyle='-',label='CCI')
    plt.plot(normalize(momentum),'r',lw=0.75,linestyle='-',label='Momentum')

    plt.legend(loc=4,prop={'size':9})
    plt.ylabel('b%')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    
    fig.subplots_adjust(hspace=0)
    plt.savefig('indicator_all_normed.png', dpi=150, format='png')
    '''


if __name__ == "__main__":
    test_run()
