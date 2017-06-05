"""
Test strategy

@author Xiaolu

"""

import datetime as dt
import pdb
import time
import traceback

import StrategyLearner1 as sl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

import market_simulator as ms
import util

# overwrite to allow testing vs. old project specs
TRADE_SIZE = 200

# how much do you have to beat another metric by to "significantly outperform" it?
OUTPERFORM = 0.1  # 10%

IN_SAMPLE_START_DATE = dt.datetime(2008, 1, 1)
IN_SAMPLE_END_DATE = dt.datetime(2009, 12, 31)
OUT_OF_SAMPLE_START_DATE = dt.datetime(2010, 1, 1)
OUT_OF_SAMPLE_END_DATE = dt.datetime(2011, 12, 31)
START_CASH = 100000

ORDERS_FNAME = 'orders.csv'
VERBOSE = False

# launch debugger on exceptions
DEBUGGER = False

# plotting parameters
PLOT_RESULTS = True  # set to True to generate performance plots
PORTFOLIO_COLOR = 'b'  # blue
BENCHMARK_COLOR = 'k'  # black
TARGET_COLOR = '#ff0000'  # red
SAVE_PLOTS = False  # True = save; False = show plots one by one
PLOT_FNAME = 'test%02d.png'

# Tests description
# Test 1 - For ML4T-220, the trained policy should provide a cumulative return greater than 400% in sample
# Test 2 - For ML4T-220, the trained policy should provide a cumulative return greater than 100% out of sample
# Test 3 - For IBM, the trained policy should significantly outperform the benchmark in sample
# Test 4 - For SINE_FAST_NOISE, the trained policy should provide a cumulative return greater than 200% in sample
# Test 5 - For UNH, the trained policy should significantly outperform the benchmark in sample

TESTS = [
    # test number, train symbol, train start, train end, test symbol, test start, test end, target
    # target = absolute growth goal; if none then goal is to significantly outperform benchmark
    # test numbers must be unique
    [1, 'ML4T-220', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE, 'ML4T-220', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE,
     1.0],
    [2, 'ML4T-220', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE, 'ML4T-220', OUT_OF_SAMPLE_START_DATE,
     OUT_OF_SAMPLE_END_DATE, 1.0],
    [3, 'AAPL', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE, 'AAPL', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE, None],
    [4, 'SINE_FAST_NOISE', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE, 'SINE_FAST_NOISE', IN_SAMPLE_START_DATE,
     IN_SAMPLE_END_DATE, 2.0],
    [5, 'UNH', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE, 'UNH', IN_SAMPLE_START_DATE, IN_SAMPLE_END_DATE, None]]


def run_tests():
    results = pd.DataFrame(index=[test[0] for test in TESTS], columns=['return', 'target', 'status'])
    results.index.name = 'Test No.'

    for testno, train_symbol, train_start_date, train_end_date, test_symbol, test_start_date, test_end_date, target in TESTS:
        start = time.time()
        # instantiate a strategy learner
        # could probably do this just once beforehand, unless addEvidence is implemented to be truly additive (vs. destructive)
        learner = sl.StrategyLearner(verbose=VERBOSE)

        learner.addEvidence(symbol=train_symbol, sd=train_start_date, ed=train_end_date, sv=START_CASH)
        trades = learner.testPolicy(symbol=test_symbol, sd=test_start_date, ed=test_end_date, sv=START_CASH)

        # sanity check - this should be impossible
        maxposition = int(abs(trades.cumsum()).max())
        if maxposition > TRADE_SIZE:
            raise ValueError("Maximum allowable position of %d exceeded in Test %d" % (TRADE_SIZE, testno))

        # generate orders file
        with open(ORDERS_FNAME, 'w') as ofp:
            ofp.write('Date,Symbol,Order,Shares\n')  # header
            for date, delta in trades.itertuples():
                if delta == 0:
                    continue
                date = str(date).split()[0]
                if delta > 0:
                    action = 'BUY'
                else:
                    action = 'SELL'
                quantity = abs(delta)
                ofp.write('%s,%s,%s,%d\n' % (date, test_symbol, action, quantity))

        portfolio_results = ms.compute_portvalue(orders_file=ORDERS_FNAME, start_val=START_CASH,
                                                 start_date=test_start_date, end_date=test_end_date)
        portfolio_results /= portfolio_results.ix[0]  # normalize

        if target is None or PLOT_RESULTS:
            benchmark = util.get_data([test_symbol], pd.date_range(test_start_date, test_end_date))
            benchmark = benchmark[[test_symbol]]  # del benchmark['SPY']
            benchmark *= TRADE_SIZE  # account for how many shares we have (scaling from one share)
            benchmark += START_CASH - benchmark.ix[0]  # add in the cash we didn't spend on the lone transaction
            benchmark /= benchmark.ix[0]  # normalize

        cumulative_return = float(portfolio_results.ix[-1]) - 1
        if target is None:
            benchmark_cumulative_return = float(benchmark.ix[-1]) - 1
            target = benchmark_cumulative_return + OUTPERFORM

        if cumulative_return > target:
            status = "passed"
        else:
            status = "failed"

        results.loc[testno] = cumulative_return, target, status
        print("total time", str(time.time() - start))

        if PLOT_RESULTS:
            plt.plot(benchmark, BENCHMARK_COLOR)
            plt.plot(portfolio_results, PORTFOLIO_COLOR)
            plt.axhline(target + 1.0, color=TARGET_COLOR)

            # labels
            plt.title('Test %d: Strategy vs. Benchmark' % testno)
            plt.xlabel('Date')
            plt.ylabel('Normalized Performance')

            # create legend
            benchmark_line = mlines.Line2D([], [], color=BENCHMARK_COLOR, label='Benchmark')
            portfolio_line = mlines.Line2D([], [], color=PORTFOLIO_COLOR, label='Strategy')
            target_line = mlines.Line2D([], [], color=TARGET_COLOR, label='Target Returns')
            plt.legend(handles=[portfolio_line, benchmark_line, target_line], loc='upper left', fancybox=True)

            if SAVE_PLOTS:
                plt.savefig(PLOT_FNAME % testno)
                plt.close()
            else:
                plt.show()
                plt.close()

    print results


if __name__ == '__main__':
    try:
        run_tests()
    except:
        if DEBUGGER:
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise
