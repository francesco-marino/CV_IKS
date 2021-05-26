# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:18:31 2021

@author: alberto
"""

import numpy as np
from findiff import FinDiff

from Problem import Problem, quickLoad
from Solver import Solver
from Orbitals import ShellModelBasis
from Energy import interpolate
from scipy import integrate
from Constants import coeffSch
import matplotlib.pyplot as plt

# loading file like u.dat and f.dat
def loadUF(file):
    
    file = open(file)
    ff = file.readlines()
    file.close()
    R = []; U = []; orb = []
    for j,ll in enumerate(ff):
        if (str(ll).startswith("#") and j==0) :
            orb.append(ll)
            r = []; u = []
        elif ( str(ll).startswith("#") ):
            orb.append(ll)
            R.append(r)
            U.append(u)
            r = []; u = []
        else:
            ll = [ float(x) for x in ll.split() ]
            r.append( ll[0] )
            u.append( ll[1] )

    R.append(r)
    U.append(u)     
    R=np.array(R); U=np.array(U)
    return (R,U,orb)    


# kinetic calculations with u 
def getKineticU(nucl, r, u):
    sigma = 0
    l, j, deg = nucl.orbital_set.getLJD()
    for i in range(len(nucl.orbital_set)):
        f = d_dx(u[i])**2 + l[i]*(l[i]+1)/r[i]**2 * u[i]**2
        sigma += deg[i] * integrate.simpson(f, r[i])
    
    return coeffSch*sigma

# NB: watch out the step!
d_dx  = FinDiff(0, 0.1, 1, acc=4)
# d2_dx  = FinDiff(0, 0.1, 2, acc=4)

# defining problem
nucl = Problem(Z=8,N=8,max_iter=4000, ub=9., debug='y', basis=ShellModelBasis(),\
               data=quickLoad("Densities/rho_o16_t0t3.dat"),\
               constr_viol=1e-4, output_folder="O16n_t0t3_for_orbitals", rel_tol=1e-4)
res, info = nucl.solve()

solver = Solver(nucl)
solver.solve()

# loading IKS u
name = "Results/O16n_t0t3_for_orbitals/u.dat"
r,u,orb = loadUF(name)

# post diagonalisation
temp = u[-1]
u[-1] = u[-2]
u[-2] = temp
u = solver.eigenvectors

K_IKS = getKineticU(nucl, r, u)

    
# loading HF u 
r_HF=[]; u_HF=[]
r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1s12.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1p32.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1p12.dat")
r_HF.append(r_temp); u_HF.append(u_temp)

K_HF = getKineticU(nucl, r_HF, u_HF)

print("\nComputed kinetic with: \n IKS ", K_IKS, "\n HF ", K_HF)
    


# plotting u
for j in range(u.shape[0]):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(r[j], u[j], label=nucl.orbital_set[j].name+r" IKS Python")
    ax.plot(r_HF[j], u_HF[j], ls='--', label=nucl.orbital_set[j].name+" HF" )
    ax.legend(); ax.grid()
    ax.set_title(r"u(r) for $\lambda=1$ ")
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])
    
# plotting u differences
for j in range(u.shape[0]):
    u_fun = interpolate(r[j], u[j])
    u_fun_HF = interpolate(r_HF[j], u_HF[j])
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(r[j], u_fun(r[j])-u_fun_HF(r[j]), label=nucl.orbital_set[j].name+r" IKS-HF difference")
    ax.legend(); ax.grid()
    ax.set_title(r"$\Delta$u(r) for $\lambda=1$ ")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\Delta$u(r)")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])
    
#%% Ca40 n t0t3

# defining problem
nucl = Problem(Z=20,N=20,max_iter=4000, ub=11., debug='y', basis=ShellModelBasis(),\
               data=quickLoad("Densities/rho_ca40_t0t3.dat"),\
               constr_viol=1e-4, output_folder="Ca40n_t0t3_for_orbitals", rel_tol=1e-4)
res, info = nucl.solve()


# loading HF u 
r_HF=[]; u_HF=[]
r_temp, u_temp = quickLoad("HF_orbitals/Ca40n_t0t3_1s12.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/Ca40n_t0t3_1p32.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/Ca40n_t0t3_1p12.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/Ca40n_t0t3_1d52.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/Ca40n_t0t3_2s12.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/Ca40n_t0t3_1d32.dat")
r_HF.append(r_temp); u_HF.append(u_temp)

print(getKineticU(nucl, r_HF, u_HF))