# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:29:13 2020

@author: jingh
"""
import numpy as np



class KalmanFilter():
    def __init__(self, A, B, C, R, Q):
        # The motion model is Linear & Gaussion
        # x_t = A*x_t_1 + B_t * u_t + w_t
        self.A = A
        self.B = B
        # The noise w ~ N(0, R)
        self.R = R
        # Dimension of state
        self.n = self.A.shape[0]
                
        # The sensor model is Linear & Gaussian
        self.C = C
        # The noise delta ~ N(0, Q)
        self.Q = Q
        # # Dimension of observation
        # self.m = self.Q.shape[0]
        
        
        
    def kf_etimation(self, x_t_1, sigma_t_1, u_t, z_t, A=None, B=None, C=None, R=None, Q=None):
        # self.check_time_variance(A, B, C, Q, R)
        
        # Kalman filter estimation
        x_t = self.A*x_t_1 + self.B*u_t
        sigma_t = self.A*sigma_t_1*self.A.T + self.R
        
        K_t = sigma_t * self.C.T * np.linalg.inv(self.C*sigma_t*self.C.T + self.Q)
        x_t = x_t + K_t*(z_t - self.C*x_t)
        sigma_t = (np.identity(self.n) - K_t*self.C) * sigma_t
        
        return x_t, sigma_t
    


if __name__ == '__main__':
    # A 1-D example in lecture notes
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    
    R = np.array([[1, 0], [0, 1]])
    Q = np.array([[1, 0], [0, 1]])
    
    kf = KalmanFilter(A, B, C, R, Q)
    
    # Initial belief
    x = np.array([[0], [0]])
    sigma = np.array([[1, 0], [0, 1]])
    u = np.array([[1], [1]])
    z = np.array([[2], [0.8]])
    
    x1, sigma1 = kf.kf_etimation(x, sigma, u, z)
    
    
    
    
    
        
    
    
    
        














        