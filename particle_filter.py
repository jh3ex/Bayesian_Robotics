# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:50:22 2020

Bayesian Robotics course project:
    modificed particle filter algorithm


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
        
        
    
        

        
    
    

















