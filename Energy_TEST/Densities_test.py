# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:03:20 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns

"""
Plotting (normalized) densities for different volume radii
"""

def plot(R, f, title):
    
    fig,ax=plt.subplots(figsize=(10,6))
    
    sns.set_theme(style='whitegrid',palette='tab20')
    
    for i in range(len(R)):
        sns.lineplot(
            x=R[i], y=f(R[i]),
            ax=ax,
            label = "R="+str('%.0f'%R[i][-1]),
            )
    ax.set_xlabel("volume radius")
    ax.set_ylabel("densities")
    ax.set_title(title)

#normalized density
rho_norm = lambda r : rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)

a = 5.
R=[]
for i in range(15):
    r = np.arange(5.,i+6.,0.001)
    R.append(r)

## Density = constant

rho = lambda r : a * np.ones_like(r)

plot(R, rho_norm, "CONSTANT")

## Density = sin^2

rho = lambda r : np.sin(r)**2

plot(R, rho_norm, "SIN^2")

## Density = Fermi function

a = 50.
RR = 10
aa = 0.5

rho = lambda r : a * (1+np.exp((r-RR)/aa))**-1

plot(R, rho_norm, "FERMI")