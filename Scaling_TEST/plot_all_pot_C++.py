# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:42:37 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

# from Problem import Problem, quickLoad
# from Solver import Solver
from Energy import Energy, quickLoad


prec="_-8"
name = "Ca40t0t3"
 
energy = Energy(data=quickLoad("Densities/rho_ca40_t0t3.dat"), C_code=True, \
                param_step=0.001, t_min=0.9, t_max=1.0, \
                input_dir="Scaled_Potentials/"+name+prec, scaling='l')

energy.solver()
v = np.array(energy.vL)
r = energy.R

fig, ax = plt.subplots(1,1,figsize=(5,5))
for i in range(len(v)):
    ax.plot(
        r, v[i,:],
        lw = 2
        )

plt.grid(); #plt.legend()
ax.set_title("Potentials for all lambdas")
ax.set_xlabel("Radius r")
# ax.set_xlim([0, 12])
# ax.set_ylim([-50, -100])
ax.set_ylabel("Potential v")

"""
integrand = np.array(energy.integrand)
fig, ax = plt.subplots(1,1,figsize=(5,5))
for i in range(len(v)):
    ax.plot(
        r, integrand[i,:],
        lw = 2
        )

plt.grid(); #plt.legend()
ax.set_title("Integrand function for all lambdas")
ax.set_xlabel("Radius r")
ax.set_xlim([0, 12])
# ax.set_ylim([-50, -100])
ax.set_ylabel("Integrand values")
"""