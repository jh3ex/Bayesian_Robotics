# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:43:00 2020

1D Kalman Filter


@author: jingh
"""

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D():
    def __init__(self, A, B, C, R, Q):
        # The motion model is Linear & Gaussion
        # x_t = A*x_t_1 + B_t * u_t + w_t
        self.A = A
        self.B = B
        # The noise w ~ N(0, R)
        self.R = R

        # The sensor model is Linear & Gaussian
        self.C = C
        # The noise delta ~ N(0, Q)
        self.Q = Q
        
        
    def kfe(self, x_t_1, sigma_t_1, u_t, z_t):
        # Kalman filter estimation
        x_t = self.A*x_t_1 + self.B*u_t
        sigma_t = self.A*sigma_t_1*self.A + self.R
        
        K_t = sigma_t * self.C * 1/(self.C*sigma_t*self.C + self.Q)
        x_t = x_t + K_t*(z_t - self.C*x_t)
        sigma_t = (1 - K_t*self.C) * sigma_t
        
        return x_t, sigma_t
        
        


if __name__ == '__main__':
    # A 1-D example in lecture notes
    A, B, C = 1, 1, 1
    R = 1
    Q = 1
    kf = KalmanFilter1D(A, B, C, R, Q)
    
    # Initial belief
    x = 0
    sigma = 1
    u = [1, 1, 1, 1]
    z = [1.43, 1.71, 3.18, 4.42]
    
    for i in range(len(z)):
        x, sigma = kf.kfe(x, sigma, u[i], z[i])
        print(x, sigma)
        
    
    
        
    
    
    
        














        