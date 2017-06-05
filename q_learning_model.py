"""
Q-learning model, with an optional dyna-Q feature.

@author Xiaolu
"""

import random as rand

import numpy as np


class QLearner(object):
    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.Q = np.random.uniform(-1, 1, (num_states, num_actions))
        if dyna > 0:
            self.visited = {}
            self.Tc = 0.00001 * np.ones((num_states, num_actions, num_states))
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)  # T: prob of (s,a) end up with s'
            self.R = -1 * np.ones((num_states, num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        rand_val = rand.random()
        if rand_val >= self.rar:
            action = np.argmax(self.Q[s])
        else:
            action = rand.randint(0, self.num_actions - 1)

        self.s = s
        self.a = action

        if self.verbose: print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """

        # update Q table
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + \
                                 self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :]))

        # dyna, lecture's algorithm using T, R tables
        if self.dyna > 0:
            # keep track of visited (s,a) pairs
            if self.s not in self.visited:
                self.visited[self.s] = set()
            self.visited[self.s].add(self.a)

            self.Tc[self.s, self.a, s_prime] += 1
            self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / self.Tc[self.s, self.a, :].sum()
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            # count: how many experiences ((s,a,s') pairs) we've learned
            count = (self.Tc >= 1).sum()
            if count >= 500:  # start dyna after 500 experiences
                for i in range(self.dyna):
                    # s: randomly chosen from previously visited states
                    s_rand = np.random.choice(list(self.visited.keys()))
                    # a: randomly chosen from previously taken actions for that random s you just chose
                    a_rand = np.random.choice(tuple(self.visited[self.s]))
                    # choose s_prime_dyna based on probability/weight by infer T[s_rand, a_rand, :]
                    s_prime_dyna = np.random.choice(range(self.num_states), p=self.T[s_rand, a_rand, :])
                    # r: R[s,a]               
                    r_dyna = self.R[s_rand, a_rand]
                    # update Q
                    self.Q[s_rand, a_rand] = (1 - self.alpha) * self.Q[s_rand, a_rand] + \
                                             self.alpha * (r_dyna + self.gamma * np.max(self.Q[s_prime_dyna, :]))

        # update s_prime and action_prime
        action_prime = self.querysetstate(s_prime)  # updated s and a within this function
        # update rar
        self.rar = self.rar * self.radr

        if self.verbose: print "s =", s_prime, "a =", action_prime, "r =", r
        return action_prime


if __name__ == "__main__":
    print "Q-learning"
