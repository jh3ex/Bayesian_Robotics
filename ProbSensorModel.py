# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:38:30 2020

Bayesian Robotics course project:
    Probabilistic sensor model

@author: jingh
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import copy


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
        
    
    
    
    
if __name__ == '__main__':
    
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
    
    
    