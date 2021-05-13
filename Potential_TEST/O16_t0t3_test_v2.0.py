# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:55:03 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from Misc import read
from Problem import Problem
from Orbitals import ShellModelBasis
from Energy import Energy, scaleDensityFun, interpolate, floatCompare
from Plot_templ import Plot


def getCutoff(rho):
        for rr in np.arange(0.,50.,0.1):
            if( rho(rr) < 1e-8 ):
                return rr - 0.1


def getTheoreticalPotentialScaled_t0t3(r, rho_p, rho_n, t):
    #parameters
    t0 = -2552.84; t3=16694.7; alfe=0.20309
    
    rho_p_fun = interpolate(r, rho_p)
    rho_n_fun = interpolate(r, rho_n)
    
    rho_p_fun = scaleDensityFun(rho_p_fun, t, 'l')
    rho_n_fun = scaleDensityFun(rho_n_fun, t, 'l')
    
    r_max = getCutoff(rho_n_fun)
    r = np.arange(0.,r_max+0.1,0.1)
    # """
    first = t0/4.+t3/24.*(rho_p_fun(r)+rho_n_fun(r))**alfe
    second = t3/24.*alfe*(rho_p_fun(r)+rho_n_fun(r))**(alfe-1)
    third = (rho_p_fun(r)+rho_n_fun(r))**2 + 2*rho_p_fun(r)*rho_n_fun(r)
    
    vp = first * (2*rho_p_fun(r)+4*rho_n_fun(r)) + second * third
    vn = first * (2*rho_n_fun(r)+4*rho_p_fun(r)) + second * third
    """
    vp = 3./2.*t0*rho_p_fun(r)+\
        t3/4.*(2*rho_p_fun(r))**alfe*rho_p_fun(r)+\
            t3/4.*alfe*(2*rho_p_fun(r))**(alfe-1)*rho_p_fun(r)**2
    vn = vp
    """
    ### OK, they are the same as long as the scaling is applied to both densities
    return r, vp, vn


def getTheoreticalEnergyScaled_t0t3(r, rho_p, rho_n, t):
    #parameters
    t0 = -2552.84; t3=16694.7; alfe=0.20309
    
    rho_p_fun = interpolate(r, rho_p)
    rho_n_fun = interpolate(r, rho_n)
    
    rho_p_fun = scaleDensityFun(rho_p_fun, t, 'l')
    rho_n_fun = scaleDensityFun(rho_n_fun, t, 'l')
    
    r_max = getCutoff(rho_n_fun)
    r = np.arange(0.,r_max+0.1,0.1)
    
    integrand = ((rho_p_fun(r)+rho_n_fun(r))**2+2*rho_p_fun(r)*rho_n_fun(r))\
                    *(t0/4.+t3/24.*(rho_p_fun(r)+rho_n_fun(r))**alfe)
    E = integrate.simpson(4*np.pi*r**2*integrand, r)

    return E

def getTheoreticalKinetic_t0t3(t):
    # HF value for total kinetic energy
    # O16
    T =  148.20364636580385 *2 
    
    return t**2*T


particle = 'n'

# loading datas
dat=read("Densities/rho_o16_t0t3.dat")
rho_p = dat[1]
rho_n = dat[2]
nucl_name = "O16"+particle+"_t0t3"
if particle == 'p': 
    dat= dat[0],dat[1]
    rho_p_fun = interpolate(dat[0], dat[1])
else:
    dat= dat[0],dat[2]
    rho_n_fun = interpolate(dat[0], dat[1])
    
#%% COMPUTING POTENTIALS

nucl = Problem(Z=8,N=8, n_type='p', max_iter=4000, ub=10., debug='y', \
                basis=ShellModelBasis(), data=dat, \
                output_folder=nucl_name+"_graphs", exact_hess=True)

lambdas = np.arange(0.5, 1.5, 0.1)

# for Python
energy = Energy(problem=nucl,  \
                param_step=0.1, t_max=lambdas[0], t_min=1.0, \
                scaling='l', output="Output", cutoff=1e-8)

#for C++
energy_C = Energy(data=dat, C_code=True, \
                  param_step=0.01, t_min=1, t_max=lambdas[0], \
                  input_dir="Scaled_Potentials/" + nucl_name + "_-8", scaling='l')


v_prot = []; v_neut = []; r = []
v_IKS = []; r_IKS = []
r_C=[]; v_C=[]
for i,l in enumerate(lambdas):
    energy.setNewParameters(t=l)
    energy.getPotential_En()
    
    energy_C.setNewParameters(t=l) 
    energy_C.getPotential_En()
    
    r_temp, v1, v2 = getTheoreticalPotentialScaled_t0t3(dat[0], rho_p, rho_n, l)
    v_prot.append(v1)
    v_neut.append(v2)
    r.append(r_temp)
    if l<1:
        v_IKS.append(energy.vL[0])
        r_IKS.append(energy.rL[0])
        
        v_C.append(energy_C.vL[0])
        r_C.append(energy_C.rL[0])
    else: 
        v_IKS.append(energy.vL[-1])
        r_IKS.append(energy.rL[-1])
        
        v_C.append(energy_C.vL[-1])
        r_C.append(energy_C.rL[-1])
        
# plotting potentials
for i in range(len(v_IKS)):
   fig, ax = plt.subplots(1,1,figsize=(5,5))
   ax.plot(
       r_IKS[i], v_IKS[i],
       color = "blue",
       label = "IKS Python",
       lw=2
       )
   ax.plot(
       r_C[i], v_C[i] - v_C[i][55] + v_IKS[i][55],
       color = "red",
       label = "IKS C++",
       lw=2
       )
   if particle == 'p':
       ax.plot(
           r[i], v_prot[i] - v_prot[i][55] + v_IKS[i][55],
           color = "orange",
           label = "Theoretical",
           lw = 2
       )
   else:
       ax.plot(
           r[i], v_neut[i] - v_neut[i][55] + v_IKS[i][55],
           color = "orange",
           label = "Theoretical",
           lw = 2
       )
   plt.grid(); plt.legend()
   ax.set_title("Scaled potentials for lambda= " +str(lambdas[i])+" "+nucl_name)
   ax.set_xlabel(r"$\lambda$")
   ax.set_xlim([0, 11])
   # ax.set_ylim([-5, 0])
   ax.set_ylabel("Potential")
   
#%% COMPUTING ENERGIES

nucl = Problem(Z=8,N=8, n_type='p', max_iter=4000, ub=10., debug='y', \
                basis=ShellModelBasis(), data=dat, \
                output_folder=nucl_name+"_graphs", exact_hess=True)

lambdas = np.arange(0.5, 1.5, 0.1)

# for Python
energy = Energy(problem=nucl,  \
                param_step=0.1, t_max=lambdas[0], t_min=1.0, \
                scaling='l', output="Output", cutoff=1e-8)

#for C++
energy_C = Energy(data=dat, C_code=True, \
                  param_step=0.01, t_min=1, t_max=lambdas[0], \
                  input_dir="Scaled_Potentials/" + nucl_name + "_-8", scaling='l')
    
    
U = np.zeros_like(lambdas)
K=[];

U_theor = np.zeros_like(lambdas)
K_theor = np.zeros_like(lambdas)

U_C = np.zeros_like(lambdas)
K_C = []
for i,l in enumerate(lambdas):
    energy.setNewParameters(t=l)
    U[i]=energy.getPotential_En()
    
    energy_C.setNewParameters(t=l) 
    U_C[i]=energy_C.getPotential_En()
    
    if floatCompare(l, energy.T_L):
        temp1, temp2 = energy.getKinetic_En()
        temp3, temp4 = energy_C.getKinetic_En()
        if l<1:
            K.append(temp2[0])
            K_C.append(temp4[0])
            
        else: 
            K.append(temp2[-1])
            K_C.append(temp4[-1])
            
    U_theor[i] = getTheoreticalEnergyScaled_t0t3(dat[0], rho_p, rho_n, l)
    K_theor[i] = getTheoreticalKinetic_t0t3(l)

U_real= getTheoreticalEnergyScaled_t0t3(dat[0], rho_p, rho_n, 1.)
K_real= getTheoreticalKinetic_t0t3(1.)

# plotting singular energies 
"""
Plot(lambdas, U, r"$\lambda$", "Potential difference", \
     "Energy differences Python "+nucl_name) 
    

Plot(lambdas, U_theor-U_real, r"$\lambda$", "Potential difference", \
     "Energy differences Theoretical "+nucl_name)
    
Plot(lambdas, K, r"$\lambda$", "Kinetic", \
     "Kinetic energies C++ Python "+nucl_name) 

Plot(lambdas, K_theor, r"$\lambda$", "Theoretical kinetic", \
     "Theoretical kinetic energies "+nucl_name)
    
Plot(lambdas, U_C, r"$\lambda$", "Kinetic", \
     "Kinetic energies C++ "+nucl_name)
"""

### NOTE:
    # in the following plots a *2 appears, it should take into account both the \
        # protonic and neutronic energy for IKS procedure.
    
# kinetic energy differences
#NB, K[5] and K_C[5] are the real kinetic energies; must change if lambdas is changed
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    lambdas, np.array(K)*2 - K[5]*2,
    color = "blue",
    label = "IKS PYTHON",
    lw=2
    )
ax.plot(
    lambdas, np.array(K_C)*2 - 2* K_C[5],
    color = "red",
    label = "IKS C++",
    lw=2
    )
ax.plot(
    lambdas, K_theor - K_real,
    color = "orange",
    label = "Theoretical",
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("Kinetic energy differences "+nucl_name)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-5, 0])
ax.set_ylabel("Kinetic enegy difference")
    
# potential energy differences
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    lambdas, U*2,
    color = "blue",
    label = "IKS Python",
    lw=2
    )
ax.plot(
    lambdas, U_C*2,
    color = "red",
    label = "IKS C++",
    lw=2
    )
ax.plot(
    lambdas, U_theor-U_real,
    color = "orange",
    label = "Theoretical",
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("Potential energy differences "+nucl_name)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-5, 0])
ax.set_ylabel("Potential energy difference")


#%% PLOTTING TOTAL ENERGY DIFFERENCES

# IKS PYTHON
Plot(lambdas, U+K-K[5], r"$\lambda$", "total energy difference", \
     "IKS PYTHON total energy difference "+nucl_name)
    
# IKS C++
Plot(lambdas, U_C+K_C-K_C[5], r"$\lambda$", "total energy difference", \
     "IKS C++ total energy difference "+nucl_name)
    
# Theoretical
Plot(lambdas, U_theor-U_real+K_theor-K_real, r"$\lambda$", "total energy difference", \
     "Theoretical total energy difference "+nucl_name)
    
    
# plotting them all together: 
"""
# total energy differences
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    lambdas, (U+K-K[5])*2,
    color = "blue",
    label = "IKS Python",
    lw=2
    )
ax.plot(
    lambdas, (U_C+K_C-K_C[5])*2,
    color = "red",
    label = "IKS C++",
    lw=2
    )
ax.plot(
    lambdas, U_theor-U_real+K_theor-K_real,
    color = "orange",
    label = "Theoretical",
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("Potential energy differences "+nucl_name)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-5, 0])
ax.set_ylabel("Potential energy difference")
"""   
