'''
GaussianEdgeMarkov is a directed graph object where each edge is a normal random variable.
Each edge will be represented by a tuple (mu, sigma) where mu is the mean (statistic) weight and sigma is the standard deviation.
'''

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs


class Graph():
    def __init__(self, mat):
        self.mat = mat
    
    def add_edge(self, u, v, weight):
        if self.exists_edge(u, v):
            # raise Exception('Edge already exists')
            pass
        self.mat.loc[u, v] = weight

    def remove_edge(self, u, v):
        if not self.exists_edge(u, v):
            # raise Exception('Edge does not exist')
            pass
        self.mat.loc[u, v] = 0

    def get_edge(self, u, v):
        return self.mat.loc[u, v]

    def get_edge_by_idx(self, i, j):
        return self.mat.iloc[i, j]

    def get_stationary_dist(self):
        mat = self.mat.to_numpy()
        mat = np.transpose(mat)
        val, vec = eigs(mat, which='LM', k=1)
        vec = np.ndarray.flatten(abs(vec))
        return vec
    
    def exists_edge(self, i, j):
        return self.mat.loc[i, j] != 0
    

class GaussianEdgeMarkov(Graph):
    def __init__(self, mat):
        self.mat = mat.astype('object')

    def add_edge(self, u, v, mu, sigma):
        if self.exists_edge(u, v):
            pass
            # raise Exception('Edge already exists')
        # self.mat.loc[u][v] = (mu, sigma)
        self.mat.loc[u][v] = (mu, sigma)
    
    def add_to_edge(self, u, v, mu, sigma):
        self.mat.loc[u][v] = (self.get_edge_mean(u, v) + mu, self.get_edge_std(u, v) + sigma)

    def remove_edge(self, u, v):
        if not self.exists_edge(u, v):
            raise Exception('Edge does not exist')
        self.mat.loc[u, v] = 0
    
    def get_edge(self, u, v):
        if not self.exists_edge(u, v):
            raise Exception('Edge does not exist')
        return self.mat.loc[u, v]
    
    def get_edge_mean(self, u, v):
        return self.get_edge(u, v)[0]
    
    def get_edge_std(self, u, v):
        return self.get_edge(u, v)[1]

    def monte_carlo_sample(self, n=1000):
        '''
        Returns a list of n samples from the graph.
        '''
        samples = []
        for _ in range(n):
            samples.append(self.one_sample())
        return samples
    
    def one_sample(self):
        '''
        Returns a single sample from the graph.
        '''
        n = self.mat.shape[0]
        sample = []
        for i in range(n):
            u = self.mat.index[i]
            row = []
            for j in range(n):
                v = self.mat.index[j]
                if self.exists_edge(u, v):
                    row.append(np.random.normal(self.get_edge_mean(u, v), self.get_edge_std(u, v)))
                else:
                    row.append(0)
            sample.append(row)
        return Graph(pd.DataFrame(sample))

    def mc_stationary_dist(self, n=1000):
        '''
        Returns the stationary distribution of the graph.
        '''
        samples = self.monte_carlo_sample(n)
        dim = self.mat.shape[0]
        res = []
        for s in samples:
            res.append(s.get_stationary_dist())
        res = np.array(res)
        # may also just be interesting to look at the distribution of the stationary distribution, not just mean
        return np.average(res, axis=0)
        
