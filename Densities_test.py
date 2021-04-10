# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:03:20 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


a = 50.
RR = 10
aa = 0.5
rho = lambda r : a * (1+np.exp((r-RR)/aa))**-1
rho_norm = lambda r : rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)

graf=[]; graf_norm=[]
for R in np.arange(10.,20.1,1.):
    r = np.arange(0.,R,0.001)
    graf_norm.append(rho_norm(r))
    graf.append(rho(r))
    print("normalization const: R="+str(R), integrate.simpson(rho(r) * r**2 *4*np.pi, r))
    
fig, ax = plt.subplots(1,1,figsize=(5,5))
for i in range(len(graf)):
    r = np.arange(0.,i+10,0.001)
    ax.plot(
        r, graf_norm[i],  
        label = "R_max=" + str(i+10),
        color = "orange"
        )
plt.grid(); plt.legend()
ax.set_title("Density for different R_max")
ax.set_xlabel("radius r"),
ax.set_ylabel("Density")

fig, ax = plt.subplots(1,1,figsize=(5,5))
for i in range(len(graf)):
    r = np.arange(0.,i+10,0.001)
    ax.plot(
        r, graf[i],  
        label = "R_max=" + str(i+10),
        color = "blue"
        )
plt.grid(); plt.legend()
ax.set_title("Density for different R_max")
ax.set_xlabel("radius r"),
ax.set_ylabel("Density")

print(integrate.simpson(rho(r) * r**2 *4*np.pi, r))
print(integrate.simpson(4*np.pi*r**2*rho_norm(r),r))# OK 1!