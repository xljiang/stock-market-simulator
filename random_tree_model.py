"""
Random Tree Model

@author Xiaolu
"""

import numpy as np
import random
from scipy import stats


class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        pass
    
    
    def build_tree(self, data):
        # decision tree is a np array, each row is 
        # [feature, splitval, left_tree offset, right_tree offset]
        if data.shape[0] == 0:
            return []
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, stats.mode(data[:,-1])[0][0], -1, -1]]) # feature == -1 means leaf
        if len(np.unique(data[:,-1])) == 1:
            return np.array([[-1, stats.mode(data[:,-1])[0][0], -1, -1]])
        else:
            feature = random.randint(0, data.shape[1]-2)
            split_val = np.mean([data[random.randint(0, data.shape[0]-1), feature],\
                                data[random.randint(0, data.shape[0]-1), feature]])
            
            # resample for up to 10 times if it's a tie and max
            # if cannot resolve within 10 tries, label it a terminal leaf
            i = 0
            while (split_val == np.max(data[:,feature]) and i < 10):
                split_val = np.mean([data[random.randint(0, data.shape[0]-1), feature],\
                                data[random.randint(0, data.shape[0]-1), feature]])
                i += 1
            if (i == 10):
                return np.array([[-1, stats.mode(data[:,-1])[0][0], -1, -1]])
            
            left_tree = self.build_tree(data[data[:,feature] <= split_val])
            right_tree = self.build_tree(data[data[:,feature] > split_val])
            #print "left tree:", left_tree
            #print "left tree type:", type(left_tree)
            #print "left tree shape:", left_tree.shape
            if not np.any(left_tree): # left_tree is empty
                return np.array([[-1, stats.mode(data[:,-1])[0][0], -1, -1]])
            
            root = np.array([feature, split_val, 1, left_tree.shape[0]+1])
            return np.vstack((root, left_tree, right_tree))
            
            

    def query_single_test(self, test):
        row = 0 # start from root
        # feature: tree[row, 0]
        # split_val: tree[row, 1]
        # left offset: tree[row, 2]
        # right offset: tree[row, 3]
        while self.tree[row, 0] != -1:
            feature = self.tree[row, 0]
            if test[feature] <= self.tree[row, 1]:
                row += self.tree[row, 2] # go to the left tree, find the row number of the left child tree root
            else:
                row += self.tree[row, 3] # go to the right tree, find the row number of the right child tree root

        return self.tree[row, 1] # return split value, when feature = -1, is leaf
                                 # the split value of the leaf is the prediction


    def addEvidence(self,Xtrain,Ytrain):
        """
        @summary: Add training data to learner
        @param Xtrain: X values of data to add
        @param Ytrain: the Y training values
        """
        
        # combine Xtrain and Ytrain to one data
        data = np.column_stack((Xtrain, Ytrain.T))
      
        # build and save the model
        self.tree = self.build_tree(data)

    
    # take a set of Xtest values and returns corresponding Ypredict set    
    def query(self,Xtest):
        """
        @summary: Estimate a set of test points (Xtest) given the model we built.
        @param Xtest: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model. a set of Y values
        """

        Ypredict = np.zeros(shape=(Xtest.shape[0],))
        for i in range(0, Xtest.shape[0]): #loop through every row in Xtest
            Ypredict[i] = self.query_single_test(Xtest[i,:])

        return Ypredict


