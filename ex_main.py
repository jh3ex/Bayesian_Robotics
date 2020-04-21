# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:05:18 2020

Bayesian Robotics course project:
    Particle filter exmaple

@author: jingh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from . import ParticleFilter, ProbMotionModel, ProbSensorModel
from scipy.stats import norm



if __name__ == '__main__':
    
    np.random.seed(999)
    
    walk_rd = np.random.RandomState(seed=123)
    """
    Motion model
    """
    n = 1000
    delta_t = 1
    v_mu, v_sigma = 1.47, 0.12
    theta_dot_mu, theta_dot_sigma = 0, 0.15
    trans_prob = np.array([[0.3, 0.7], [0.1, 0.9]])
    
    pmm = ProbMotionModel(v_mu, v_sigma, theta_dot_mu, theta_dot_sigma, trans_prob, delta_t, n)
    
    
    """
    Sensor Model
    """
    x_s, y_s, theta_s = -1, 4, 0.2
    fov = 1.7
    dmax = 10
    z_mu = np.array([0, 0])
    z_cov = np.array([[0.05, 0], [0, 0.005]])
    
    
    psm = ProbSensorModel(fov, dmax, z_mu, z_cov)

    """
    Particle Filter
    """
    pf = ParticleFilter(n, pmm, psm)
    
    
    """
    Initialization
    """
    x0_mu, x0_sigma = 0, 0.1
    y0_mu, y0_sigma = 5, 0.1
    theta0_mu, theta0_sigma = 0.2, 0.02
    
    # Initial particles
    xt = pmm.initial_particles(x0_mu, x0_sigma, y0_mu, y0_sigma, theta0_mu, theta0_sigma)
    # Initial state
    x, y, theta = pmm.initial_sample(x0_mu, x0_sigma, y0_mu, y0_sigma, theta0_mu, theta0_sigma, walk_rd)
    
    # Recursive particle filter estimation
    plt.figure(dpi=300)
    
    err_set = []
    
    """
    Recursive estimation
    """
    
    for k in range(30):
        
        # Make observation
        depth, beta, zx, zy = psm.observe(x, y, x_s, y_s, theta_s, walk_rd)
        
        xtm1 = copy.deepcopy(xt)
        
        # # Particle filter estimation
        xt = pf.pf_one_step(xt, depth, beta, x_s, y_s, theta_s)
        
        _, _, err = pf.evaluation(xt, x, y)
        
        err_set.append(err)
        
        if k >= 1:
            xt_theta, d_walk = pf.re_orientation(xtm1, xt)
            
        #     # xt.theta = norm.rvs(1, 0.001, size=n)
            # Orientation correction
            if d_walk > 0.6:
                # xt.theta = norm.rvs(xt_theta, 0.001, size=n)
                xt.theta = xt_theta
        
        # Plot
        # plt.figure(dpi=300)
        plt.scatter(xt.x, xt.y, s=1)
        plt.scatter([x, zx], [y, zy], c=['b', 'r'])
        
        plt.scatter(x_s, y_s, marker='^', c=['g'])
        # # plt.title('step %d || err %g' % (k+1, err))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title('Step %d || Error: %g' % (k, err))
        # plt.show() 
        
  
        
        # Make another move
        x, y, theta = pmm.walk_one_step(x, y, theta)
        
        
        # Sensor move
        x_s += 1.2
        y_s += 0.1
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('30 Steps || Mean Error: %g' % (np.mean(err_set)))
    plt.show()    
    print(np.mean(err_set))
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    