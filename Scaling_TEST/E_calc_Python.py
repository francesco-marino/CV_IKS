# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:10:29 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Orbitals import ShellModelBasis 
from Problem import Problem
# from Solver import Solver
from Energy import Energy, quickLoad
from Misc import read

prec = "_-8"
n = 0

if n==0:
    nucl_name = "O16p_SkX"
    data = quickLoad("Densities/SkXDensityO16p.dat")
    Z=8; N=8
if n==1:
    nucl_name = "O16n_SkX"
    data = quickLoad("Densities/SkXDensityO16n.dat")
    Z=8; N=8
elif n==2:
    nucl_name = "O16t0t3"
    data = quickLoad("Densities/rho_o16_t0t3.dat")
    Z=8; N=8
elif n==3:
    nucl_name = "Ca40t0t3"
    data = quickLoad("Densities/rho_ca40_t0t3.dat")
    Z=8; N=8
elif n==4:
    nucl_name = "Ca40t0t3_coul/Ca40t0t3_01"
    data = quickLoad("Densities/rho_ca40_t0t3_coul.dat")
    Z=20; N=20
elif n==5:
    nucl_name = "Ca40t0t3_coul/Ca40t0t3_001"
    data = quickLoad("Densities/rho_ca40_t0t3_coul.dat")
    Z=20; N=20
elif n==6:
    nucl_name = "Ca40t0t3"
    data = quickLoad("Densities/rho_ca40_t0t3.dat")
    Z=20; N=20
elif n==7:
    nucl_name = "Ca40pSkX"
    data = quickLoad("Densities/SkXDensityCa40p.dat")
    Z=20; N=20
elif n==8:
    nucl_name = "Ca40n_SkX"
    data = quickLoad("Densities/SkXDensityCa40n.dat")
    Z=20; N=20
elif n==9: 
    nucl_name = "Ca40n_t0t3"
    data = read("Densities/rho_ca40_t0t3.dat")
    data = data[0], data[2]
    Z=20; N=20
    
#defining problem
nucl = Problem(Z,N, max_iter=2000, rel_tol=1e-4, constr_viol=1e-4, \
                  data=data,
                  basis=ShellModelBasis(), exact_hess=True)
    
    
l_min = np.arange(0.9,1.,0.01)
if abs(l_min[-1]-1.)<1e-6: l_min=np.delete(l_min, -1)
E_sx = np.zeros_like(l_min)
dEdL_sx = []; grid=[]

energy = Energy(problem=nucl,  \
                param_step=0.001, t_min=l_min[0], t_max=1.0, \
                scaling='l', output=nucl_name+prec)

for i,l in enumerate(l_min):
    energy.setNewParameters(t_m=l)
    E_sx[i]=energy.solver()
    gridL, dE = energy.dEdL()
    dEdL_sx.append(dE)
    grid.append(gridL)
    
    
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    l_min, E_sx,
    lw = 2,
    # s = 5
    )
plt.grid(); #plt.legend()
ax.set_title("Energy differences O16p_SkX_-8")
ax.set_xlabel("Lambda bound")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("Energy difference")

for i in range(len(grid)):
    dEdL_sx[i] = np.array(dEdL_sx[i]) # * -1. 
    dEdL_sx[i] = np.reshape(dEdL_sx[i], -1)
    
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    grid[0], dEdL_sx[0],
    lw = 2
    )
plt.grid(); #plt.legend()
ax.set_title("dE/dL O16p_SkX_-8")
ax.set_xlabel("Lambda")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("dE/dL")

grid_sx = grid[0]
###############################################################################

l_max = np.arange(1.01,1.11,0.01)
if abs(l_max[-1]-1.11)<1e-6: l_max=np.delete(l_max, -1)
# print(l_max)
E_dx = np.zeros_like(l_max)
dEdL_dx = []; grid=[]

energy = Energy(problem=nucl, \
                param_step=0.001, t_min=1.0, t_max=l_max[0], \
                output=nucl_name+prec, scaling='l')


for i,l in enumerate(l_max):
    energy.setNewParameters(t_M=l)
    E_dx[i]=energy.solver()
    
    gridL, dE = energy.dEdL()
    dEdL_dx.append(dE)
    grid.append(gridL)
    

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    l_max, E_dx,
    lw = 2
    )
plt.grid(); #plt.legend()
ax.set_title("Energy differences " + nucl_name + prec)
ax.set_xlabel("Lambda bound")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("Energy difference")

for i in range(len(grid)):
    dEdL_dx[i] = np.array(dEdL_dx[i])
    dEdL_dx[i] = np.reshape(dEdL_dx[i], -1)

# print(grid, dEdL_sx)
grid_dx = grid[-1]
fig, ax = plt.subplots(1,1,figsize=(5,5))
for i in range(len(grid)):
    ax.plot(
        grid[i], dEdL_dx[i],
        lw = 2
        )
plt.grid(); #plt.legend()
ax.set_title("dE/dL " + nucl_name + prec)
ax.set_xlabel("Lambda")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("dE/dL")

###############################################################################
#%% plotting energy differences and derivatives together
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    l_min, E_sx,
    l_max, E_dx,
    color = "blue",
    lw = 2
    )
plt.grid(); #plt.legend()
ax.set_title("Energy differences " + nucl_name + prec)
ax.set_xlabel("Lambda bound")
# ax.set_xlim([0, 11])
# ax.set_ylim([-5, 0])
ax.set_ylabel("Energy difference")
print(grid[0])
fig, ax1 = plt.subplots(1,1,figsize=(5,5))
ax1.plot(
    grid_sx, dEdL_sx[0],
    grid_dx, dEdL_dx[-1],
    color="blue",
    lw = 2
    )
plt.grid(); #plt.legend()
ax1.set_title("dE/dL " + nucl_name + prec)
ax1.set_xlabel("Lambda")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax1.set_ylabel("dE/dL")
#%% TEST PLOT (for n=2!!) (even for n=8 (almost))
integral=[]
# print(test[0], test[1])
for i in range(len(test[0])):
    if i!=0:
        integral.append(energy.integrator(test[1][0:i],test[0][0:i]))

integral = np.flip(integral)   
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot( 
    test[0][1:], integral,
    lw = 1
    )
ax.plot( 
    test[0][1:], 37.8*(test[0][1:] - test[0][-1]) + integral[-1],
    color = "orange",
    lw = 1
    )
plt.grid(); #plt.legend()
ax.set_title("DeltaE")   
ax.set_xlabel("Lambda")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("DeltaE")   

from findiff import FinDiff

d_dx  = FinDiff(0, 0.01, 1, acc=4)

print(integral, d_dx(integral))
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot( 
    test[0][1:], d_dx(integral),
    lw = 1
    )
plt.grid(); #plt.legend()
ax.set_title("DeltaE")   
ax.set_xlabel("Lambda")

# ax.set_xlim([0.9, 0.99])
# ax.set_ylim([-40, -30])
ax.set_ylabel("DeltaE")   