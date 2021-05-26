# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:47:29 2021

@author: alberto
"""

import numpy as np
from findiff import FinDiff

from Misc import loadData, saveData
from Problem import Problem, quickLoad
from Orbitals import ShellModelBasis
from Energy import interpolate
from scipy import integrate
from Constants import coeffSch
import matplotlib.pyplot as plt
from Solver import Solver

def getCutoff(rho, cutoff = 1e-8):
    for rr in np.arange(0.,50.,0.1):
        if( rho(rr) < cutoff ):
            return rr - 0.1

def scaleDensityFun(rho, t, scale):
    q = lambda r: t * rho(r)
    l = lambda r: t**3 * rho(t*r)
    z = lambda r: t**2 * rho(t**(1./3.)*r) 
    
    if scale == "all" :
        return q, l, z
    if scale == "q" :
        return q
    if scale == "l" :
        return l
    if scale == "z" :
        return z


file = "Densities/rho_o16_t0t3.dat"
r_dens, dens = quickLoad(file)
rho = interpolate(r_dens, dens)
nucl_name = "O16n_t0t3"
part = 8

r=[]; v=[]; epsilon=[]; int_rhov=[]; K=[]; real_lambdas=[]

check = (b"Algorithm terminated successfully at a locally optimal point, "
             b"satisfying the convergence tolerances (can be specified by options).")

lambdas = np.arange(0.6,1.4,0.05)
for l in lambdas:
    Rho = scaleDensityFun(rho, l, 'l')
    bound = getCutoff(Rho)
    
    nucl = Problem(Z=part,N=part,max_iter=2000, ub=bound, debug='y', \
               basis=ShellModelBasis(), rho=Rho,\
               constr_viol=1e-4, output_folder=nucl_name+"_out", rel_tol=1e-4)
    print("Solving problem with L= "+str(l)+" and bound "+str(bound))
    res, info = nucl.solve()
    
    if res['status']==check:
        solver = Solver(nucl)
        solver.solve()
        
        r.append(solver.grid)
        v.append(solver.getPotential())
        
        a, b, c = solver.printAll()
        K.append( a )
        epsilon.append( b )
        int_rhov.append( c )
        real_lambdas.append( l ) 

eps_name = "Results/"+nucl_name+"_out"+"/Epsilon_sum.dat"
int_name = "Results/"+nucl_name+"_out"+"/int_rhov.dat"
saveData(eps_name, epsilon)
saveData(int_name, int_rhov)

# problem with lambda = 1
bound = getCutoff(rho)
nucl = Problem(Z=part,N=part,max_iter=2000, ub=bound, debug='y', \
               basis=ShellModelBasis(), rho=rho,\
               constr_viol=1e-4, output_folder=nucl_name+"_out", rel_tol=1e-4)
res, info = nucl.solve()

solver = Solver(nucl)
solver.solve()

r_real = (solver.grid)
v_real = (solver.getPotential())

K_real, epsilon_real, int_rhov_real = solver.printAll()



# plotting eps, int rho and kinetics
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(real_lambdas, epsilon, label=r"IKS")
ax.plot(real_lambdas, np.array(real_lambdas)**2*epsilon_real, ls='--', label=r"Theoretical")
ax.legend(); ax.grid()
ax.set_title("Epsilon sums" + nucl_name)
ax.set_xlabel("r")
ax.set_ylabel(r"$\epsilon$")  


fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(real_lambdas, int_rhov, label=r"IKS")
ax.plot(real_lambdas, np.array(real_lambdas)**2*int_rhov_real, ls='--', label=r"Theoretical")
ax.legend(); ax.grid()
ax.set_title(r"$\int dr \rho v$" + nucl_name)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\int dr \rho v$")


fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(real_lambdas, K, label=r"IKS")
ax.plot(real_lambdas, np.array(real_lambdas)**2*K_real, ls='--', label=r"IKS")
ax.legend(); ax.grid()
ax.set_title(r"Kinetics" + nucl_name)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"K")     
        

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(real_lambdas, np.array(K) + np.array(int_rhov) - np.array(epsilon), label=r"IKS")
ax.legend(); ax.grid()
ax.set_title(r"K + $\int dr \rho v$ - $\sum_i \epsilon_i$ " + nucl_name)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"K + $\int dr \rho v$ - $\sum_i \epsilon_i$ ") 


#%%
# problem with lambda = 1
bound = getCutoff(rho)
nucl = Problem(Z=part,N=part,max_iter=2000, ub=bound, debug='y', \
               basis=ShellModelBasis(), rho=rho, h=0.01,\
               constr_viol=1e-4, output_folder=nucl_name+"_out", rel_tol=1e-4)
res, info = nucl.solve()

solver = Solver(nucl)
solver.solve()

r_temp = (solver.grid)
v_temp = (solver.getPotential())

K_temp, epsilon_temp, int_rhov_temp = solver.printAll()

fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(r_temp, v_temp, label=r"IKS")
ax.legend(); ax.grid()
ax.set_title("Epsilon sums" + nucl_name)
ax.set_xlabel("r")
ax.set_ylabel(r"$\epsilon$")  
ax.set_xlim([0, 8.5])
ax.set_ylim([-50, 40])

u = solver.getDiagOrbitals()

r_HF=[]; u_HF=[]
r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1s12.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1p32.dat")
r_HF.append(r_temp); u_HF.append(u_temp)
r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1p12.dat")
r_HF.append(r_temp); u_HF.append(u_temp)

for j in range(u.shape[0]):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(solver.grid, u[j], label=solver.sorted_orbital_set[j].name+" post diag orig")
    ax.plot(r_HF[j], u_HF[j], ls='--', label=nucl.orbital_set[j].name+" HF" )
    ax.legend(); ax.grid()
    ax.set_title("Epsilon sums" + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\epsilon$")  
    # ax.set_xlim([0, 8.5])
    # ax.set_ylim([-50, 40])



#%%
d_dx  = FinDiff(0, 0.01, 1, acc=4)
# kinetic calculations with u 
def getKineticU(nucl, r, u):
    sigma = 0
    l, j, deg = nucl.orbital_set.getLJD()
    for i in range(len(nucl.orbital_set)):
        f = d_dx(u[i])**2 + l[i]*(l[i]+1)/r[i]**2 * u[i]**2
        sigma += deg[i] * integrate.simpson(f, r[i])
    
    return coeffSch*sigma


for j in range(u.shape[0]):
    u[j][0]=0
    u_fun = interpolate(solver.grid, u[j])
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(r_HF[j], u_fun(r_HF[j]) - u_HF[j], label="difference")
    ax.grid()
    ax.set_title("Differences" + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$\epsilon$")  
    ax.set_xlim([0, 10])
    ax.set_ylim([-0.002, 0.002]) 
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

R=[]
for j in range(3):
    R.append(solver.grid)
    
print(getKineticU(nucl, R, u))