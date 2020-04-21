#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:18:15 2020

Bayesian Robotics course project:
    illustrating motion model in 1D case

@author: jingh
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class MotionModel1D:
    def __init__(self, w_mean, w_sigma, u, xt_min, xt_max, n=200):
        self.w_mean = w_mean
        self.w_sigma = w_sigma
        self.u = u
        self.n = n
        self.xt = np.linspace(xt_min, xt_max, self.n)
    
        
    def initial_belief(self, x_0_mean, x_0_sigma):
        p = norm.pdf(self.xt, x_0_mean, x_0_sigma)
        return p
        
        
    def estimation(self, pm1):
        
        p = np.zeros(self.n)
        
        for i in range(self.n):
            for j in range(self.n):
                p[i] += norm.pdf(self.xt[i], self.xt[j]+self.u+self.w_mean, self.w_sigma) * pm1[j]
                
        return p * (self.xt.max() - self.xt.min()) /(self.n)
        


if __name__ == '__main__':
    
    motion = MotionModel1D(0, 0.1, 1, -1, 5)
    p0 = motion.initial_belief(0, 0.2)
    p = motion.estimation(p0)
    
    p11 = 0.9
    
    p1 = (1-p11) * p0 + p11 * p
    
    plt.figure(dpi=300)
    plt.plot(motion.xt, p0)
    plt.plot(motion.xt, p)
    plt.plot(motion.xt, p1)
    plt.legend(['$p(x_0)$', '$p(x_1;s_1=1)$', '$p(x_1)$'])
    plt.xlabel('x')
    plt.ylabel('p')
    plt.show()
    
    p = motion.estimation(p1)
    
    p2 = (1-p11) * p1 + p11 * p
    
    plt.figure(dpi=300)
    plt.plot(motion.xt, p1)
    plt.plot(motion.xt, p)
    plt.plot(motion.xt, p2)
    plt.legend(['$p(x_1)$', '$p(x_2;s_2=1)$', '$p(x_2)$'])
    plt.xlabel('x')
    plt.ylabel('p')
    plt.show()
    
    
    p = motion.estimation(p2)
    
    p3 = (1-p11) * p2 + p11 * p
    
    plt.figure(dpi=300)
    plt.plot(motion.xt, p2)
    plt.plot(motion.xt, p)
    plt.plot(motion.xt, p3)
    plt.legend(['$p(x_2)$', '$p(x_3;s_3=1)$', '$p(x_3)$'])
    plt.xlabel('x')
    plt.ylabel('p')
    plt.show()
    
    
    
    
    
    
    
    
    


