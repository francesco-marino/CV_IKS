# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:18:31 2021

@author: alberto
"""

import numpy as np
from findiff import FinDiff

from Problem import Problem, quickLoad
# from Problem_mod import Problem_mod
from Orbitals import ShellModelBasis
from scipy import integrate
from Constants import coeffSch, T
from Energy import interpolate
import matplotlib.pyplot as plt
from Solver import Solver

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


#%% ORIGINAL PROBLEM
nucl = Problem(Z=20,N=20,max_iter=4000, ub=13., debug='y', basis=ShellModelBasis(),\
               data=quickLoad("Densities/SkXDensityCa40p.dat"),\
               constr_viol=1e-4, output_folder="SkXDensityCa40p", rel_tol=1e-4)
res, info = nucl.solve()

name = "Results/"+nucl.output_folder+"/u.dat"
# R,U,orb = loadUF(name)

u = res['u']
r = res['grid']
f = res['x']
u1 = u

d_dx  = FinDiff(0, 0.1, 1, acc=4)
d2_dx  = FinDiff(0, 0.1, 2, acc=4)

def getKineticU():
    sigma = 0
    r,u,orb = loadUF(name)
    l, j, deg = nucl.orbital_set.getLJD()
    for i in range(len(nucl.orbital_set)):
        f = d_dx(u[i])**2 + l[i]*(l[i]+1)/r[i]**2 * u[i]**2
        sigma += deg[i] * integrate.simpson(f, r[i])
    
    return coeffSch*sigma
    


def getKineticF():
    sigma = 0
    r,f,orb = loadUF(name)
    C0, C1, C2 = nucl._getCFunctions()
    l, j, deg = nucl.orbital_set.getLJD()
    for i in range(len(nucl.orbital_set)):
        I = C0[i]* f[i]**2 + C1* f[i]* d_dx(f[i])+ C2 *d2_dx(f[i])*f[i]
        sigma += deg[i] * integrate.simpson(I, r[i])
    
    return -T*sigma

name = nucl.output_folder+"/u.dat"
print("Computed kinetic with u: ", getKineticU())
name = nucl.output_folder+"/f.dat"
print("Computed kinetic with f: ", getKineticF())        
    

#%% SCALED PROBLEM
data = quickLoad("Densities/SkXDensityCa40p.dat")
dens = interpolate(data[0], data[1])
t = 0.962 
cutoff = 1e-8

def scaled_dens(r, rho, L):
    return L**3 * rho(L * r)

rho = lambda r : scaled_dens(r, dens, t)

for rr in np.arange(0.,50.,0.1):
    if( rho(rr) < cutoff ):
        bound = rr - 0.1
        break

print("lambda: ", t)
nucl2 = Problem(Z=20, N=20, n_type='p', max_iter=2000, ub=bound, debug='y', \
                  basis=ShellModelBasis(), rho=rho,\
                  exact_hess=True, output_folder="SkXDensityCa40p_"+str(t),\
                  rel_tol=1e-4, constr_viol=1e-4)
results, info = nucl2.solve()

# calculating the kinetic energy from the scaling applied directly to orbitals
rXX, rescaled_u, OrbXX = loadUF("Results/SkXDensityCa40p/u.dat")
u_fun=[]; new_rescaled_u=[]

for i in range(len(rescaled_u)):
    u_fun= interpolate(nucl.grid, rescaled_u[i])
    new_rescaled_u.append(t**0.5 * u_fun(t*nucl.grid))

for j in range(rescaled_u.shape[0]):
    fig2, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(nucl.grid, rescaled_u[j], ls='--', label=nucl.orbital_set[j].name)
    ax.plot(nucl.grid, new_rescaled_u[j], label="theoretical")
    r,u,orb = loadUF(name)
    ax.plot(nucl2.grid, results['u'][j], label=nucl2.orbital_set[j].name)
    ax.legend()
    ax.set_title("u(r) for lambda=" + str(t) + " Ca40pSkX")
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    ax.grid()
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])

def getKineticU():
    sigma = 0
    # r,u,orb = loadUF(name)
    r = nucl.grid
    u = new_rescaled_u
    # r = nucl2.grid
    # u = results['u']
    l, j, deg = nucl.orbital_set.getLJD()
    for i in range(len(nucl.orbital_set)):
        f = d_dx(u[i])**2 + l[i]*(l[i]+1)/r**2 * u[i]**2
        sigma += deg[i] * integrate.simpson(f, r)
    
    return coeffSch*sigma

print("\n new u manually calculated", getKineticU())


def getKineticU():
    sigma = 0
    r,u,orb = loadUF(name)
    l, j, deg = nucl2.orbital_set.getLJD()
    for i in range(len(nucl.orbital_set)):
        f = d_dx(u[i])**2 + l[i]*(l[i]+1)/r[i]**2 * u[i]**2
        sigma += deg[i] * integrate.simpson(f, r[i])
    
    return coeffSch*sigma
    

def getKineticF():
    sigma = 0
    r,f,orb = loadUF(name)
    C0, C1, C2 = nucl2._getCFunctions()
    l, j, deg = nucl2.orbital_set.getLJD()
    for i in range(len(nucl2.orbital_set)):
        I = C0[i]* f[i]**2 + C1* f[i]* d_dx(f[i])+ C2 *d2_dx(f[i])*f[i]
        sigma += deg[i] * integrate.simpson(I, r[i])
    
    return -T*sigma

name = nucl2.output_folder+"/u.dat"
print("Computed kinetic with u: ", getKineticU())
name = nucl2.output_folder+"/f.dat"
print("Computed kinetic with f: ", getKineticF()) 


#%% PLOTTING u FUNCTIONS

#loading C++ functions
name = "Scaled_Potentials/Ca40pSkX_0.962_C++/u.dat"
rC, uC, orbC = loadUF(name)

#plotting
u2 = results['u']
for j in range(u.shape[0]):
    U = interpolate(nucl.grid, u1[j])
    fig2, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(nucl.grid, u1[j,:], ls='--', label=nucl.orbital_set[j].name)
    ax.plot(nucl2.grid, u2[j,:], label=nucl2.orbital_set[j].name)
    ax.plot(nucl.grid, t**0.5 * U(t*nucl.grid), label="theoretical")
    ax.plot(rC[j], uC[j], label=orbC[j]+" C++")
    ax.legend()
    ax.set_title("u(r) for lambda=" + str(t) + " Ca40pSkX")
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")


#%% POTENTIAL WITH NEW AND USUAL METHOD

rXX, rescaled_f, OrbXX = loadUF("Results/SkXDensityCa40p/f.dat")
f_fun=[]; new_rescaled_f=[]

for i in range(len(rescaled_f)):
    f_fun= interpolate(nucl.grid, rescaled_f[i])
    new_rescaled_f.append(f_fun(t*nucl.grid))

solver_new = Solver(nucl, new_rescaled_f)
x_new, check = solver_new.solve()

solver = Solver(nucl2)
x, check = solver.solve()

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(solver.grid, solver.getPotential(), 
        color = "blue",
        ls='--', 
        label="normal"
        )
ax.plot(solver_new.grid, solver_new.getPotential(), 
        color = "orange", 
        label= "test"
        )
plt.grid(); plt.legend()
ax.set_title("Potential with lambda " + str(t))
ax.set_xlabel("Radius r")
# ax.set_xlim([0, 12])
# ax.set_ylim([-150, -100])
ax.set_ylabel("Potential")


#%% PLOTTING f FUNCTIONS
name = "Results/SkXDensityCa40p/f.dat"
r1,f1,temp = loadUF(name)
name = "Results/SkXDensityCa40p_0.962/f.dat"
r2,f2,temp = loadUF(name)

for j in range(len(f1)):
    # print(f1, f1.shape)
    U = interpolate(nucl.grid, f1[j])
    fig2, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(nucl.grid, f1[j,:], ls='--', label=nucl.orbital_set[j].name)
    ax.plot(nucl2.grid, f2[j,:], label=nucl2.orbital_set[j].name)
    ax.plot(nucl.grid, U(t*nucl.grid), label="scaled?") 
    ax.legend(); ax.grid()
    
    