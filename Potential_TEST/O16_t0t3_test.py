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
from Solver import Solver
from Orbitals import ShellModelBasis
from Energy import Energy, scaleDensityFun, interpolate, floatCompare, quickLoad
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
    
    first = t0/4.+t3/24.*(rho_p_fun(r)+rho_n_fun(r))**alfe
    second = t3/24.*alfe*(rho_p_fun(r)+rho_n_fun(r))**(alfe-1)
    third = (rho_p_fun(r)+rho_n_fun(r))**2 + 2*rho_p_fun(r)*rho_n_fun(r)
    
    vp = first * (2*rho_p_fun(r)+4*rho_n_fun(r)) + second * third
    vn = first * (2*rho_n_fun(r)+4*rho_p_fun(r)) + second * third
    
    return r, vp, vn


def getTheoreticalEnergyScaled_t0t3(r, rho_p, rho_n, t):
    #parameters
    t0 = -2552.84; t3=16694.7; alfe=0.20309
    
    rho_p_fun = interpolate(r, rho_p)
    rho_n_fun = interpolate(r, rho_n)
    
    # rho_p_fun = scaleDensityFun(rho_p_fun, t, 'l')
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
    
# normalizations
print(integrate.simpson(4*np.pi*dat[0]**2*rho_p, dat[0]))
print(integrate.simpson(4*np.pi*dat[0]**2*rho_n, dat[0]))
print(integrate.simpson(4*np.pi*dat[0]**2*(rho_p+rho_n), dat[0]))
#%% COMPUTING POTENTIALS
nucl = Problem(Z=8,N=8, n_type='p', max_iter=4000, ub=10., debug='y', \
                basis=ShellModelBasis(), data=dat, \
                output_folder=nucl_name+"_graphs", exact_hess=True)#,\
                #rel_tol=1e-6, constr_viol=1e-6)

lambdas = np.arange(0.5, 1.5, 0.1)

v_prot = []; v_neut = []; r = []
v_IKS = []; r_IKS = []
v_C=[]; r_C=[]; K_C=[]
for l in lambdas:
    if particle=='p':
        rho = scaleDensityFun(rho_p_fun, l, 'l')
        bound = getCutoff(rho)
    if particle=='n':
        rho = scaleDensityFun(rho_n_fun, l, 'l')
        bound = getCutoff(rho)
    
    nucl.setDensity(rho=rho, ub=bound)
    datas, info = nucl.solve()

    solver = Solver(nucl)
    x, check = solver.solve()
    pot = solver.getPotential()
    rad = solver.grid
    
    v_IKS.append(pot)
    r_IKS.append(rad)
    
    r_temp, v1, v2 = getTheoreticalPotentialScaled_t0t3(dat[0], rho_p, rho_n, l)
    v_prot.append(v1)
    v_neut.append(v2)
    r.append(r_temp)
    
    if particle=='p':
        name = "Scaled_Potentials\O16p_t0t3_-8\Potentials\pot_L="\
                         + str('%.3f'%l) + "000_C++.dat"
        
        name_K = "Scaled_Potentials\O16p_t0t3_-8\Kinetics\kin_L="\
                         + str('%.3f'%l) + "000_C++.dat"
    else: 
        name = "Scaled_Potentials\O16n_t0t3_-8\Potentials\pot_L="\
                         + str('%.3f'%l) + "000_C++.dat"
                         
        name_K = "Scaled_Potentials\O16p_t0t3_-8\Kinetics\kin_L="\
                         + str('%.3f'%l) + "000_C++.dat"
                         
    temp1, temp2 = quickLoad(name, beg=3, end=3)
    r_C.append(temp1)
    v_C.append(temp2)
    with open(name_K) as f:
        K_C.append( float(f.readlines()[0]) )
    
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

#%% COMPUTING C++ POTENTIAL ENERGIES

#for C++
energy_C = Energy(data=dat, C_code=True, \
                param_step=0.01, t_min=1, t_max=lambdas[0], \
                input_dir="Scaled_Potentials/" + nucl_name + "_-8", scaling='l')

E_C = np.zeros_like(lambdas)
for i,l in enumerate(lambdas):
    energy_C.setNewParameters(t=l) #t0=0.99
    E_C[i]=energy_C.getPotential_En()

Plot(lambdas, E_C, r"$\lambda$", "Kinetic", \
     "Energy differences Python "+nucl_name)
#%% COMPUTING POTENTIALS and KINETIC ENERGIES
    ## WITH ENERGY (SAME RESULTS as before)
nucl = Problem(Z=8,N=8, n_type='p', max_iter=4000, ub=10., debug='y', \
                basis=ShellModelBasis(), data=dat, \
                output_folder=nucl_name+"_graphs", exact_hess=True)

lambdas = np.arange(0.5, 1.5, 0.1)
energy = Energy(problem=nucl,  \
                param_step=0.1, t_max=lambdas[0], t_min=1.0, \
                scaling='l', output="Output", cutoff=1e-8)
    
E = np.zeros_like(lambdas)
E_theor = np.zeros_like(lambdas)
K_theor = np.zeros_like(lambdas)
v_prot = []
v_neut = []
r = []
v_IKS = []
r_IKS = []
K=[]; lam_K=[]
for i,l in enumerate(lambdas):
    energy.setNewParameters(t=l)
    E[i]=energy.getPotential_En()
    
    if floatCompare(l, energy.T_L):
        r_temp, v1, v2 = getTheoreticalPotentialScaled_t0t3(dat[0], rho_p, rho_n, l)
        v_prot.append(v1)
        v_neut.append(v2)
        r.append(r_temp)
        if l<1:
            v_IKS.append(energy.vL[0])
            r_IKS.append(energy.rL[0])
            temp1, temp2 = energy.getKinetic_En()
            K.append(temp2[0])
            lam_K.append(temp1[0])
        else: 
            v_IKS.append(energy.vL[-1])
            r_IKS.append(energy.rL[-1])
            temp1, temp2 = energy.getKinetic_En()
            K.append(temp2[-1])
            lam_K.append(temp1[-1])
        
    E_theor[i] = getTheoreticalEnergyScaled_t0t3(dat[0], rho_p, rho_n, l)
    K_theor[i] = getTheoreticalKinetic_t0t3(l)

# plotting energies (differences)
Plot(lambdas, E, r"$\lambda$", "Potential difference", \
     "Energy differences Python "+nucl_name) 
    
E_real= getTheoreticalEnergyScaled_t0t3(dat[0], rho_p, rho_n, 1.)
Plot(lambdas, E_theor-E_real, r"$\lambda$", "Potential difference", \
     "Energy differences Theoretical "+nucl_name)
    
Plot(lam_K, K, r"$\lambda$", "Kinetic", \
     "Energy differences Python "+nucl_name) 

K_real= getTheoreticalKinetic_t0t3(1.)
Plot(lambdas, K_theor, r"$\lambda$", "Theoretical kinetic", \
     "Energy differences Python "+nucl_name)
    
# kinetic energies 
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    lam_K, np.array(K)*2 - 2* K[5],
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
    
# poetntial energies
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(
    lambdas, E,
    color = "blue",
    label = "IKS Python",
    lw=2
    )
ax.plot(
    lambdas, E_C,
    color = "red",
    label = "IKS C++",
    lw=2
    )
ax.plot(
    lambdas, E_theor-E_real,
    color = "orange",
    label = "Theoretical",
    lw = 2
    )
plt.grid(); plt.legend()
ax.set_title("Potential energy differences "+nucl_name)
ax.set_xlabel(r"$\lambda$")
# ax.set_xlim([0, 11])
# ax.set_ylim([-5, 0])
ax.set_ylabel("Potential enegy difference")
"""
for i in range(len(v_IKS)):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(
        r_IKS[i], v_IKS[i],
        color = "blue",
        label = "IKS",
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
    # ax.set_xlim([0, 11])
    # ax.set_ylim([-5, 0])
    ax.set_ylabel("Potential")
# """
    
#%% CALCULATING ENERGIES WITH THEORETICAL POTENTIALS

nucl = Problem(Z=8,N=8, n_type='p', max_iter=4000, ub=10., debug='y', \
                basis=ShellModelBasis(), data=dat, \
                output_folder=nucl_name+"_graphs", exact_hess=True)

lambdas = np.arange(0.5, 1.5, 0.1)
energy = Energy(C_code=True, data=dat,\
                param_step=0.1, t_max=lambdas[0], t_min=1.0, \
                scaling='l', input_dir=nucl_name, cutoff=1e-8)

E = np.zeros_like(lambdas)
v_prot = []; v_neut = []; r = []
v_IKS = []; r_IKS = []
for i,l in enumerate(lambdas):
    
    r_temp, v1, v2 = getTheoreticalPotentialScaled_t0t3(dat[0], rho_p, rho_n, l)
    v_prot.append(v1)
    v_neut.append(v2)
    r.append(r_temp)
    
    file_pot = "/Potentials/pot_L=" + str('%.3f'%l) + "000_C++.dat"
    if particle == 'p':
        save = np.column_stack((r[-1],v_prot[-1]))
    else:
        save = np.column_stack((r[-1],v_neut[-1]))
    np.savetxt(nucl_name+file_pot, save)
    
save_stat = np.column_stack((np.ones_like(lambdas),lambdas))
file_stat=nucl_name + "/Status.dat"
np.savetxt(file_stat, save_stat)

for i,l in enumerate(lambdas):
    energy.setNewParameters(t=l)
    E[i]=energy.getPotential_En()
    
E_real= getTheoreticalEnergyScaled_t0t3(dat[0], rho_p, rho_n, 1.)
Plot(lambdas, E, r"$\lambda$", "Potential energy difference", \
     "IKS Potential energy differences Python "+nucl_name)
 
E_theor=np.zeros_like(lambdas)
for i,l in enumerate(lambdas):
    E_theor[i] = getTheoreticalEnergyScaled_t0t3(dat[0], rho_p, rho_n, l)
    
Plot(lambdas, E_theor-E_real, r"$\lambda$", "Potential energy difference", \
     "Theoretical potential energy difference "+nucl_name)
# """
#%%     TOTAL ENERGY DIFFERENCE

#IKS PYTHON
Plot(lambdas, E+K-K[5], r"$\lambda$", "total energy difference", \
     "IKS PYTHON total energy difference "+nucl_name)
    
#IKS C++
Plot(lambdas, E_C+K_C-K_C[5], r"$\lambda$", "total energy difference", \
     "IKS C++ total energy difference "+nucl_name)
    
#Theoretical
Plot(lambdas, E_theor-E_real+K_theor-K_real, r"$\lambda$", "total energy difference", \
     "Theoretical total energy difference "+nucl_name)



