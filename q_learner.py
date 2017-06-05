"""
Training and Learning function for Q-learning model

@author: Xiaolu
"""

import datetime as dt
import time

import pandas as pd

import q_learning_model as ql
import util as ut


class StrategyLearner(object):
    # constructor
    def __init__(self, verbose=False):
        self.verbose = verbose

    """Training Function"""

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 1, 1), \
                    sv=10000):

        start_time = time.time()

        self.learner = ql.QLearner(100, 3, 0.2, 0.9, 0.99, 0.999, 0, False)

        # get prices and indicators
        prices, indicators = self.get_prices_and_indicators(14, sd, ed, symbol)

        # get discrized indicators dataframe
        disc_indicators, self.bins = self.discretize_training(indicators)

        # converge parameters 
        is_converged = False

        while not is_converged:
            # get initial state (disc_indicator[0])
            state = disc_indicators[0]
            # initial action
            action = self.learner.querysetstate(state)

            curr_holding = 0  # how many shares currently holds (can only be 0,200,-200)

            for day in range(1, indicators.shape[0]):
                # implement action: 0-do nothing, 1-long, 2-short
                # calculate rewards

                daily_ret = prices.ix[day, symbol] / prices.ix[day - 1, symbol] - 1

                if action == 1:  # long
                    if curr_holding > 0:
                        r = daily_ret
                    elif curr_holding < 0:
                        r = 2 * daily_ret
                    else:
                        r = daily_ret
                    curr_holding = 200

                elif action == 2:  # short
                    if curr_holding > 0:
                        r = -2 * daily_ret
                    elif curr_holding < 0:
                        r = -daily_ret
                    else:
                        r = -daily_ret
                    curr_holding = -200

                else:  # action = 0, do nothing
                    if curr_holding > 0:
                        r = daily_ret
                    elif curr_holding < 0:
                        r = -daily_ret
                    else:
                        r = 0

                # update state and action
                state = disc_indicators[day]
                action = self.learner.query(state, r)

            # set converged when time = 20sec
            # can also set max_iterations, min_iterations, and then is_converged if 
            # 40 iterations have same cummulative returns
            total_time = time.time() - start_time
            if total_time > 20:
                is_converged = True

    """Testing Function"""

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):

        # prices and indicators
        prices, indicators = self.get_prices_and_indicators(14, sd, ed, symbol)

        # get discretized indicators dataframe using bins defined in training part
        disc_indicators = self.discretize_testing(indicators, self.bins)

        # initialize trades dataframe
        trades = prices.copy()
        trades.ix[:, :] = 0

        # initialize state, action and curr_holding

        curr_holding = 0  # how many shares currently holds (can only be 0,200,-200)

        # iterate each day to fill trades dataframe
        for day in range(indicators.shape[0]):
            # get state and action: 0-do nothing, 1-long, 2-short
            state = disc_indicators[day]
            action = self.learner.querysetstate(state)
            # update curr_holding and trades dataframe based on curr_holding, and action
            trade = 0

            if action == 1:  # long
                if curr_holding > 0:
                    trade = 0
                elif curr_holding < 0:
                    trade = 400
                else:
                    trade = 200
                curr_holding = 200

            elif action == 2:  # short
                if curr_holding > 0:
                    trade = -400
                elif curr_holding < 0:
                    trade = 0
                else:
                    trade = -200
                curr_holding = -200

            else:  # do nothing
                pass
                # keep curr_holding, no update

            trades.ix[day, symbol] = trade

        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices
        return trades

    """
    Discretize Functions
        while training: return discretized indicators and bins (us pd.qcut)
        while testing: only return discretized indicators. discretize use bins defined by training (use pd.cut)
    """

    # discretize indicators for training indicators
    def discretize_training(self, indicators):
        # adjust the upper/lower bin limits to accommodate.
        bbp_disc, bbp_bin = pd.qcut(indicators['BBP'], 10, retbins=True, labels=False)
        sma_disc, sma_bin = pd.qcut(indicators['PRICE_SMA'], 10, retbins=True, labels=False)

        # combine discretized indicators
        indi_disc = (sma_disc * 10 + bbp_disc).astype(int)

        # set bin extremes to -inf and +inf
        bbp_bin[0] = sma_bin[0] = -float('inf')
        bbp_bin[-1] = sma_bin[-1] = float('inf')
        # combine bins to one 
        bins = [bbp_bin, sma_bin]
        # print bins
        return indi_disc, bins

    # discretize for testing indicators
    def discretize_testing(self, indicators, bins):
        bbp_disc = pd.cut(indicators['BBP'], bins=bins[0], labels=False)
        sma_disc = pd.cut(indicators['PRICE_SMA'], bins=bins[1], labels=False)

        # combine discretized indicators
        indi_disc = (sma_disc * 10 + bbp_disc).astype(int)
        return indi_disc

    """Process Data Functions"""

    # return prices and indicators at given data ranges
    def get_prices_and_indicators(self, window, sd, ed, symbol):

        lookback = window
        # set temp start date ahead of sd (off = window + delta)
        temp_sd = sd - dt.timedelta(days=30)
        temp_dates = pd.date_range(temp_sd, ed)

        # Read Prices
        prices = ut.get_data([symbol], temp_dates)  # only have trading days use SPY as ref
        prices = prices.ix[:, 1:]  # remove SPY

        ### Calculate SMA-14 over the entire period in a single step.
        sma = pd.rolling_mean(prices, window=lookback, min_periods=lookback)
        # sma ratio 
        smaR = prices / sma

        ### Calculate Bollinger Bands (14 day) over the entire period.
        rolling_std = pd.rolling_std(prices, window=lookback, min_periods=lookback)
        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)
        # bollinger bands percent
        bbp = (prices - bottom_band) / (top_band - bottom_band)

        ### Bring df back to the correct dates
        dates = pd.date_range(sd, ed)
        prices_temp = ut.get_data([symbol], dates)  # only for get correct start date
        first_date = prices_temp.index[0]

        prices = prices.ix[first_date:]
        smaR = smaR.ix[first_date:]
        bbp = bbp.ix[first_date:]

        indicators = pd.concat([bbp, smaR], keys=['BBP', 'PRICE_SMA'], axis=1)
        # should I normalize indicators?

        return prices, indicators


if __name__ == "__main__":
    print "Q learning Learner"
