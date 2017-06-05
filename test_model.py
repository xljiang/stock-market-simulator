"""
Test machine learning models

@author Xiaolu
"""

import math

import matplotlib.pyplot as plt
import numpy as np

import bootstrap_aggr as bl
import random_tree_model as rt


def gen_data(data):
    # shuffle data
    np.random.shuffle(data)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    # test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    print type(trainY)
    print trainY
    return trainX, trainY, testX, testY


# for a defined leaf_size, train iter_times, input data
# return mean rmse both in sample and out sample
def decision_tree_learner(leaf_size, data, iter_times):
    in_sample_res = []
    out_sample_res = []
    for i in range(0, iter_times):
        # get data
        trainX, trainY, testX, testY = gen_data(data)
        # RTLearner training
        learner = rt.RTLearner(leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)
        # in sample
        predY = learner.query(trainX)
        in_sample = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        in_sample_res.append(in_sample)
        # out sample
        predY = learner.query(testX)
        out_sample = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        out_sample_res.append(out_sample)

    return np.mean(in_sample_res), np.mean(out_sample_res)  # average rmse for a special leaf size


# for a defined bag_size and leaf_size, train iter_times, input data
# return mean rmse both in sample and out sample
def bag_learner(bag_size, leaf_size, data, iter_times):
    in_sample_res = []
    out_sample_res = []
    for i in range(0, iter_times):
        # get data
        trainX, trainY, testX, testY = gen_data(data)
        # BagLearner training
        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, \
                                bags=bag_size, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY)
        # in sample
        predY = learner.query(trainX)
        in_sample = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        in_sample_res.append(in_sample)
        # out sample
        predY = learner.query(testX)
        out_sample = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        out_sample_res.append(out_sample)

    return np.mean(in_sample_res), np.mean(out_sample_res)  # average rmse for a s


def plot_results(title, df, testset, x_label, y_label, min_index, filename, overfit=True, yaxes=[0.0, 0.1]):
    ax = df.plot(title=title, fontsize=12, linewidth=1, legend=True, figsize=[8, 6])
    if overfit: ax.axvline(x=min_index, color='red', lw=2.0, ls='--')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_ylim(yaxes)
    plt.savefig(filename, dpi=150, format='png')
