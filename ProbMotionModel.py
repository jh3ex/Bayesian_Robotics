# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:38:31 2020

Bayesian Robotics course project:
    Probabilistic motion model

@author: jingh
"""


import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import copy


class ProbMotionModel:
    def __init__(self, v_mu, v_sigma, theta_dot_mu, theta_dot_sigma, trans_prob, delta_t, n):
        self.delta_t = delta_t
        self.n = n
        self.v_mu = v_mu
        self.v_sigma = v_sigma
        self.theta_dot_mu = theta_dot_mu
        self.theta_dot_sigma = theta_dot_sigma
        self.trans_prob = trans_prob
        
        self.p_walk = self.bernoulli_approx()
        
        self.n_walk = int(self.p_walk*self.n)
    
    def bernoulli_approx(self):
        e = np.inf
        pnm1 = copy.deepcopy(self.trans_prob)
        while e > 1e-5:
            pn = np.dot(pnm1, self.trans_prob)
            e = abs(pn - pnm1).sum()
            pnm1 = copy.deepcopy(pn)
        
        p = np.dot(np.array([0, 1]), pn)[1]
        
        return p
    
    
    def initial_particles(self, x0_mu, x0_sigma, y0_mu, y0_sigma, theta0_mu, theta0_sigma):
        
        x0 = norm.rvs(loc=x0_mu, scale=x0_sigma, size=self.n)
        y0 = norm.rvs(loc=y0_mu, scale=y0_sigma, size=self.n)
        theta0 = norm.rvs(loc=theta0_mu, scale=theta0_sigma, size=self.n)
        
        xt = pd.DataFrame({'x': x0,
                           'y': y0,
                           'theta': theta0,
                           'weight': np.zeros(self.n)})
        
        return xt
    
    
    def prediction(self, xtm1):
        
        xt = copy.deepcopy(xtm1)
        
        idx_walk = np.random.choice(self.n, size=self.n_walk, replace=False)
        
        theta_dot = norm.rvs(loc=self.theta_dot_mu, scale=self.theta_dot_sigma, size=self.n_walk)
        v = norm.rvs(loc=self.v_mu, scale=self.v_sigma, size=self.n_walk)
        
        xt['theta'].loc[idx_walk] += self.delta_t*theta_dot 
        xt['x'].loc[idx_walk] += self.delta_t*v*np.cos(xt['theta'].loc[idx_walk])
        xt['y'].loc[idx_walk] += self.delta_t*v*np.sin(xt['theta'].loc[idx_walk])
        
        return xt
    
    
    def initial_sample(self, x0_mu, x0_sigma, y0_mu, y0_sigma, theta0_mu, theta0_sigma, walk_rd=None):
        
        self.walk_rd = walk_rd
        
        x = norm.rvs(loc=x0_mu, scale=x0_sigma, random_state=self.walk_rd)
        y = norm.rvs(loc=y0_mu, scale=y0_sigma, random_state=self.walk_rd)
        theta = norm.rvs(loc=theta0_mu, scale=theta0_sigma, random_state=self.walk_rd)
        
        self.walking = 1
        
        return x, y, theta
        
    
    def walk_one_step(self, x, y, theta):
        
        # Markov Chain
        self.walking = self.walk_rd.choice([0, 1], p=self.trans_prob[self.walking, :])
        
        # Bernoulli
        # self.walking = np.random.choice([0, 1], p=[1-self.p_walk, self.p_walk])
        
        if self.walking == 1:
            
            theta_dot = norm.rvs(loc=self.theta_dot_mu, scale=self.theta_dot_sigma, random_state=self.walk_rd)
            theta += theta_dot * self.delta_t

            v = norm.rvs(loc=self.v_mu, scale=self.v_sigma, random_state=self.walk_rd)
            
            x += v * np.cos(theta) * self.delta_t
            y += v * np.sin(theta) * self.delta_t
            
        return x, y, theta
            

def scatter_plot(xt):
    plt.figure(dpi=300)
    plt.scatter(xt['x'], xt['y'], s=1)
    plt.xlim([-1, 10])
    plt.ylim([-1, 10])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    n = 1000
    delta_t = 1
    v_mu, v_sigma = 1.47, 0.12
    theta_dot_mu, theta_dot_sigma = 0, 0.02
    trans_prob = np.array([[0.3, 0.7], [0.1, 0.9]])
    
    pmm = ProbMotionModel(v_mu, v_sigma, theta_dot_mu, theta_dot_sigma, trans_prob, delta_t, n)
    
    
    x0_mu, x0_sigma = 0, 0.1
    y0_mu, y0_sigma = 5, 0.1
    theta0_mu, theta0_sigma = 0.2, 0.02
    
    xt = pmm.initial_particles(x0_mu, x0_sigma, y0_mu, y0_sigma, theta0_mu, theta0_sigma)
    
    scatter_plot(xt)
    for i in range(6):
        xt = pmm.prediction(xt)
        scatter_plot(xt)
    
    
    