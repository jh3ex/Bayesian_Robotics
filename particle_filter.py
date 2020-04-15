# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:50:22 2020

@author: jingh
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import copy



class ParticleFilter:
    def __init__(self, n, pmm, psm):
        # n is particle size
        self.n = n
        self.pmm = pmm
        self.psm = psm
        
    def pf_one_step(self, xt, depth, beta, x_s, y_s, theta_s):
        
        xt = self.pmm.prediction(xt)
        xt = self.psm.correction(depth, beta, x_s, y_s, theta_s, xt)
        xt = self.resampling(xt)
        
        return xt
        
    
    def re_orientation(self, xtm1, xt):
        
        
        
        xm1_mean = xtm1.x.mean()
        ym1_mean = xtm1.y.mean()
        
        x_mean = xt.x.mean()
        y_mean = xt.y.mean()
        
        d_walk = np.sqrt((x_mean-xm1_mean)**2 + (y_mean-ym1_mean)**2)
        
        xt_theta = np.arctan((y_mean-ym1_mean) / (x_mean-xm1_mean))
        
        # xt.theta = np.arctan((xt.y-xtm1.y) / (xt.x-xtm1.x))
        
        return xt_theta, d_walk
        
    
    def resampling(self, xt):
        
        p = xt['weight']/xt['weight'].sum()
        
        idx = np.random.choice(self.n, size=self.n, replace=True, p=p)
        
        xt = xt.loc[idx]
        
        xt = xt.reset_index(drop=True)
        
        return xt
    
    
    def evaluation(self, xt, x, y):
        x_mean = xt.x.mean()
        y_mean = xt.y.mean()
        
        err = np.sqrt((x-x_mean)**2 + (y-y_mean)**2)
        
        return x_mean, y_mean, err
        
        
    
        
class ProbSensorModel:
    def __init__(self, fov, dmax, z_mu, z_cov):
        self.fov = fov  # Field of view
        self.dmax = dmax  # Extreme distance
        self.z_mu = z_mu  # Mean of observation noises
        self.z_cov = z_cov  # Covariance matrix
    
    
    def correction(self, depth, beta, x_s, y_s, theta_s, xt):
        """
        Calculate likelihood given an observation
        Arguments:
            depth:  observed depth
            beta:  observed bearing angle
            x_s, y_s, theta_s:  sensor framework
            xt:  true state
        Return:
            xt with updated weights
        """
        
        psg, psg_inv = self.transform_matrix(x_s, y_s, theta_s)
        
        
        # From true state x to depth, beta
        z = np.array([depth, beta])
        
        for i in range(xt.count()[0]):
            xt_s = np.dot(psg_inv, np.array([xt['x'].loc[i], xt['y'].loc[i], 1]))
            
            depth_hat = np.sqrt(xt_s[0]**2 + xt_s[1]**2)
            beta_hat = np.arctan(xt_s[0]/xt_s[1])
            
            mean = np.array([depth_hat, beta_hat]) + self.z_mu
            
            # Likelihood            
            p_zx = multivariate_normal.pdf(z, mean, self.z_cov)
            
            # Update weight
            xt['weight'].loc[i] = p_zx
        
        return xt
    
    
    def transform_matrix(self, x_s, y_s, theta_s):
        psg = np.array([[np.cos(theta_s), -np.sin(theta_s), x_s],
                        [np.sin(theta_s), np.cos(theta_s), y_s],
                        [0, 0, 1]])
        
        psg_inv = np.linalg.inv(psg)
        
        return psg, psg_inv
        
        

    def observe(self, x, y, x_s, y_s, theta_s, walk_rd):
        # x, y is true state
        psg, psg_inv = self.transform_matrix(x_s, y_s, theta_s)
        
        xt_s = np.dot(psg_inv, np.array([x, y, 1]))
        
        depth_hat = np.sqrt(xt_s[0]**2 + xt_s[1]**2)
        beta_hat = np.arctan(xt_s[0]/xt_s[1])
        
        # Observation noise
        v_noise = multivariate_normal.rvs(mean=self.z_mu, cov=self.z_cov, random_state=walk_rd)
        
        depth = depth_hat + v_noise[0]
        beta = beta_hat + v_noise[1]
        
        z_s = np.array([depth*np.sin(beta), depth*np.cos(beta), 1])
        
        z_g = np.dot(psg, z_s)
        
        zx, zy = z_g[0], z_g[1]
        
        return depth, beta, zx, zy
        
        
    
    
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
    
    
    # x, y, theta = pmm.initial_sample(x0_mu, x0_sigma, y0_mu, y0_sigma, theta0_mu, theta0_sigma)
    
    # # Sensor Model
    # fov = 1.7
    # dmax = 10
    # z_mu = np.array([0, 0])
    # z_cov = np.array([[0.2, 0], [0, 0.05]])
    
    # psm = ProbSensorModel(fov, dmax, z_mu, z_cov)
    
    # x_s, y_s, theta_s = -1, 4, 0
    # depth, beta, zx, zy = psm.observe(x, y, x_s, y_s, theta_s)
    
    # plt.figure(dpi=300)
    # plt.scatter([x, zx], [y, zy], c=['b', 'r'])
    # plt.show()
















