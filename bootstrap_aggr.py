"""
Bootstrap Aggregation - bagging

@author Xiaolu
"""

import numpy as np
from scipy import stats

import random_tree_model as rt


class BagLearner(object):
    def __init__(self, learner=rt.RTLearner, kwargs={"leaf_size": 1}, \
                 bags=20, boost=False, verbose=False):
        self.learners = []
        self.bags = bags
        self.boost = boost
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

        pass

    def addEvidence(self, Xtrain, Ytrain):
        for i in range(0, self.bags):
            sampling_index = np.random.choice(len(Ytrain), len(Ytrain), replace=True)
            Xtrain_sampling = Xtrain[sampling_index, :]
            Ytrain_sampling = Ytrain[sampling_index]
            self.learners[i].addEvidence(Xtrain_sampling, Ytrain_sampling)

    def query(self, Xtest):
        Ypredict = np.zeros(shape=(Xtest.shape[0], self.bags))
        for i in range(0, self.bags):
            Ypredict[:, i] = self.learners[i].query(Xtest)
        return stats.mode(Ypredict, axis=1)[0].T[0]  # return a 1d np array
