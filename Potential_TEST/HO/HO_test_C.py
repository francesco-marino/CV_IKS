# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:25:37 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
from Problem import quickLoad
from scipy.integrate import simpson

dataC = quickLoad("Potentials/pot_HO_126_particles_C++.dat")
dataP = quickLoad("Potentials/pot_HO_126_particles_coupled_basis_pyt.dat")
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    dataC[0], dataC[1],
    color = "orange",
    label = "CV C++", 
    lw = 2
    )
ax.plot(
    dataC[0], 1.2*dataC[0]**2 + dataC[1][0],
    color = "blue",
    ls = '--',
    label = "quadratic", 
    lw = 2
    )
ax.plot(
    dataP[0], dataP[1] + dataC[1][0] - dataP[1][0],
    color = "green",
    label = "CV python", 
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("HO potential with n=20")
ax.set_xlabel("Radius r")
ax.set_xlim([0, 15])
ax.set_ylabel("Potential")
#%%
dataC = quickLoad("Potentials/pot_HO_20_particles_C++.dat")
dataP = quickLoad("Potentials/pot_HO_20_particles_coupled_basis_pyt.dat")
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    dataC[0], dataC[1],
    color = "orange",
    label = "CV C++", 
    lw = 2
    )
ax.plot(
    dataC[0], 2.8*dataC[0]**2 + dataC[1][0],
    color = "blue",
    ls = '--',
    label = "quadratic", 
    lw = 2
    )
ax.plot(
    dataP[0], dataP[1] + dataC[1][0] - dataP[1][0],
    color = "green",
    label = "CV python", 
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("HO potential with n=20")
ax.set_xlabel("Radius r")
ax.set_xlim([0, 15])
ax.set_ylabel("Potential")
#%%
data = quickLoad("Densities/SkXDensityCa40p.dat")
print("0: \t", simpson(4*np.pi*data[1]*data[0]**2,data[0]))

data = quickLoad("temp.dat.txt")
print("1: \t", simpson(4*np.pi*data[1]*data[0]**2,data[0]))
#if I want it normalized then data[0]/lambda (real r and NOT lambda * r) is needed

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    data[0], data[1],
    color = "orange",
    label = "CV", 
    lw = 2
    )
plt.grid(); plt.legend()
# ax.set_title("HO potential with n="+str(n))
ax.set_xlabel("Radius r")
ax.set_xlim([0, 15])
ax.set_ylabel("Density")

data = quickLoad("datas.txt")
print("2: \t", simpson(4*np.pi*data[1]*data[0]**2,data[0]))
#CORRECTLY NORMALIZED!!