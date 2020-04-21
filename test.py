# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:15:56 2020

Bayesian Robotics course project:
    Orientation correction study

@author: jingh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def mse(truth, sample):
    
    se = 0
    
    
    
    for s in sample:
        se += (s - truth)**2
    
    return np.mean(se)
        
    



if __name__ == '__main__':
    
    n = 50
    
    x1 = norm.rvs(1, 1, n)
    y1 = norm.rvs(1, 1, n)
    
    x2 = norm.rvs(4, 1, n)
    y2 = norm.rvs(4, 1, n)
    
    plt.figure(dpi=300)
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.show()
    
    theta1 = []
    
    for i in range(n):
        for j in range(n):
            theta1.append(np.arctan((y2[i]-y1[j]) / (x2[i]-x1[j])))


    theta1 = np.array(theta1)    

    print(np.degrees(theta1.mean()))
    print(mse(np.pi/4, theta1))
    
    
    theta2 = []
    
    for i in range(n):
        theta2.append(np.arctan((y2[i]-y1[i]) / (x2[i]-x1[i])))

    theta2 = np.array(theta2)    

    print(np.degrees(theta2.mean()))
    print(mse(np.pi/4, theta2))

    theta3 = np.arctan((y2.mean()-y1.mean()) / (x2.mean()-x1.mean()))
    print(np.degrees(theta3))
    print((theta3-np.pi/4)**2)
    
    
    
    
    
    
    
    