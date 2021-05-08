# -*- coding: utf-8 -*-
"""
Created on Tue May  4 00:12:37 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Plot_templ import Plot
from Orbitals import ShellModelBasis 
from Problem import Problem
from Energy import Energy, quickLoad, floatCompare
from Misc import read

prec = "_-8"
n = 2.5

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
elif n==2.5:
    nucl_name = "O16n_t0t3"
    data = read("Densities/rho_o16_t0t3.dat")
    data = data[0], data[2]
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
                  data=data, basis=ShellModelBasis(), exact_hess=True)
    
###############################################################################

l_min = np.arange(0.3,1.,0.01)
if abs(l_min[-1]-1.)<1e-6: l_min=np.delete(l_min, -1)

#for python
energy_P = Energy(problem=nucl,  \
                param_step=0.01, t_max=l_min[0], t_min=1.0, \
                scaling='l', output=nucl_name+prec, cutoff=1e-8)
#for C++
energy_C = Energy(data=data, C_code=True, \
                param_step=0.01, t_max=l_min[0], t_min=1.0, \
                input_dir="Scaled_Potentials/" + nucl_name + prec, scaling='l')
    
E_P_sx = np.zeros_like(l_min)
E_C_sx = np.zeros_like(l_min)
for i,l in enumerate(l_min):
    energy_P.setNewParameters(t=l)
    E_P_sx[i]=energy_P.getPotential_En()
    energy_C.setNewParameters(t=l)
    E_C_sx[i]=energy_C.getPotential_En()
    
    if i==0: 
        grid_K_C_sx, K_C_sx = energy_C.getKinetic_En()
        grid_K_P_sx, K_P_sx = energy_P.getKinetic_En()
        K_true = K_P_sx[-1]
        grid_P_sx, dE_P_sx = energy_P.dEdt()
        grid_C_sx, dE_C_sx = energy_C.dEdt()
       
# plotting energies (differences)
Plot(l_min, E_P_sx, r"$\lambda$", "Potential difference", \
     "Energy differences Python "+nucl_name+prec) 
Plot(l_min, E_C_sx, r"$\lambda$", "Potential difference", \
     "Energy differences C++ "+nucl_name+prec) 
# plotting derivatives
# dE_P_sx_sign = [ -x for x in dE_P_sx]
Plot(grid_P_sx, dE_P_sx, r"$\lambda$", "dE/dL", \
     "dE/dL Python "+nucl_name+prec)
# dE_C_sx_sign = [ -x for x in dE_C_sx]
Plot(grid_C_sx, dE_C_sx, r"$\lambda$", "dE/dL", \
     "dE/dL C++ "+nucl_name+prec)
 
###############################################################################
#defining problem
nucl = Problem(Z,N, max_iter=2000, rel_tol=1e-4, constr_viol=1e-4, \
               data=data, basis=ShellModelBasis(), exact_hess=True)
    
l_max = np.arange(1.01,1.7,0.01)
if abs(l_max[-1]-1.7)<1e-6: l_max=np.delete(l_max, -1)

#for python
energy_P = Energy(problem=nucl, \
                param_step=0.01, t_min=1, t_max=l_max[0], \
                output=nucl_name+prec, scaling='l', cutoff=1e-8)
#for C++
energy_C = Energy(data=data, C_code=True, \
                param_step=0.01, t_min=1, t_max=l_max[0], \
                input_dir="Scaled_Potentials/" + nucl_name + prec, scaling='l')

E_P_dx = np.zeros_like(l_max)
E_C_dx = np.zeros_like(l_max)
for i,l in enumerate(l_max):
    # print("looking for it")
    energy_P.setNewParameters(t=l) #t0=0.99
    E_P_dx[i]=energy_P.getPotential_En()
    
    energy_C.setNewParameters(t=l) #t0=0.99
    E_C_dx[i]=energy_C.getPotential_En()

    if i==( len(l_max)-1) :
        grid_K_C_dx, K_C_dx = energy_C.getKinetic_En()
        grid_K_P_dx, K_P_dx = energy_P.getKinetic_En()
        grid_P_dx, dE_P_dx = energy_P.dEdt()
        grid_C_dx, dE_C_dx = energy_C.dEdt()
  
# plotting energies (differences)
# E_P_dx_sign = [ -x for x in E_P_dx]
Plot(l_max, E_P_dx, r"$\lambda$", "Potential difference", \
     "Energy differences Python "+nucl_name+prec) 
# E_C_dx_sign = [ -x for x in E_C_dx]
Plot(l_max, E_C_dx, r"$\lambda$", "Potential difference", \
     "Energy differences C++ "+nucl_name+prec)
# plotting derivatives
# dE_P_dx_sign = [ -x for x in dE_P_dx]
Plot(grid_P_dx, dE_P_dx, r"$\lambda$", "dE/dL", \
     "dE/dL Python "+nucl_name+prec)
# dE_C_dx_sign = [ -x for x in dE_C_dx]
Plot(grid_C_dx, dE_C_dx, r"$\lambda$", "dE/dL", \
     "dE/dL C++ "+nucl_name+prec)

###############################################################################

# plotting energy differences and derivatives together
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    l_min, E_P_sx,
    l_max, E_P_dx,
    color = "blue",
    label = "python",
    lw=2
    )
ax.plot(
    l_min, E_C_sx,
    l_max, E_C_dx,
    color = "orange",
    label = "C++",
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("Energy differences " + nucl_name + prec)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-5, 0])
ax.set_ylabel("Energy difference")
# print(grid[0])

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    grid_P_sx, dE_P_sx,
    grid_P_dx, dE_P_dx,
    color="blue",
    label = "python",
    lw=2
    )
ax.plot(
    grid_C_sx, dE_C_sx,
    grid_C_dx, dE_C_dx,
    color="orange",
    label = "C++",
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("dE/dL " + nucl_name + prec)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("dE/dL")

# plotting kinetic energies (not differences!) and their theoretical behaviour
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    grid_K_P_sx, K_P_sx,
    grid_K_P_dx, K_P_dx,
    color = "blue",
    label = "IKS",
    lw=2
    )
ax.plot(
    grid_K_C_sx, K_C_sx,
    grid_K_C_dx, K_C_dx,
    color = "red",
    label = "C++",
    lw=2
    )
ax.plot(
    grid_K_P_sx, K_true*grid_K_P_sx**2,
    grid_K_P_dx, K_true*grid_K_P_dx**2,
    color = "orange",
    ls='--',
    label = "theoretical",
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("Kinetic energies " + nucl_name + prec)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("K")

##kinetic differences (C++)
real_K_sx = np.ones_like(K_C_sx)*K_C_sx[-1] 
real_K_dx = np.ones_like(K_C_dx)*K_C_sx[-1]
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    np.concatenate((grid_K_C_sx, grid_K_C_dx)),
    np.concatenate((K_C_sx - real_K_sx, K_C_dx - real_K_dx)),
    color = "red",
    label = "C++",
    lw=2
    )
plt.grid(); plt.legend()
ax.set_title("Kinetic differences " + nucl_name + prec)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel("K")


# plotting total energy differences
## C++

b, l = quickLoad("Scaled_Potentials/" + nucl_name + prec + "/Status.dat")
elim = np.where(b!=1)
for e in elim[0]:
    if floatCompare(l[e], l_min):
        rem = np.where(abs(l[e]-l_min)<1e-6)
        l_min = np.delete(l_min, rem)
        E_C_sx = np.delete(E_C_sx, rem)
    if floatCompare(l[e], l_max):
        rem = np.where(abs(l[e]-l_max)<1e-6)
        l_max = np.delete(l_max, rem)
        E_C_dx = np.delete(E_C_dx, rem)
        
real_K_sx = np.ones_like(E_C_sx)*K_C_sx[-1]
real_K_dx = np.ones_like(E_C_dx)*K_C_sx[-1]
K_C_sx_graph = np.delete(K_C_sx, -1)
K_C_dx_graph = np.delete(K_C_dx, 0)

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    l_min, E_C_sx + K_C_sx_graph - real_K_sx,
    l_max, E_C_dx + K_C_dx_graph - real_K_dx, 
    color = "blue",
    label = "C++",
    lw=2
    )
plt.grid(); plt.legend()
ax.set_title("Total energy differences " + nucl_name + prec)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel(r"$\Delta$ E_tot")

""" ## must check and reshape sizes
## PYTHON
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    l_min, E_P_sx + K_P_sx - K_true,
    l_max, E_P_dx + K_P_dx - K_true, 
    color = "blue",
    label = "IKS",
    lw=2
    )
plt.grid(); plt.legend()
ax.set_title("Total energy differences " + nucl_name + prec)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-50, -100])
ax.set_ylabel(r"$\Delta$ E_tot")
"""
