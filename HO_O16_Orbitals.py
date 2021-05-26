# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:26:07 2021

@author: alberto
"""

import numpy as np

from Constants import nuclearNu, nuclearOmega, m_p
from Misc import read
from Problem import Problem
from Orbitals import ShellModelBasis
from scipy import integrate
from Energy import interpolate, scaleDensityFun, quickLoad
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


########### ORIGINAL PROBLEM
n = 0
if n==0:
    part = 20
    U_name = "Scaled_Potentials/HO_20_particles_L=0.62/u.dat"
    file = "Densities/HO/rho_HO_"+str(part)+"_particles_coupled_basis.dat"
    nucl_name = "HO_"+str(part)+"_particles"
elif n==1: 
    part = 8
    U_name = "Scaled_Potentials/O16n_t0t3_L=0.6/u.dat"
    file = "Densities/rho_o16_t0t3.dat"
    nucl_name = "O16n_t0t3"
elif n==2: 
    part = 20
    U_name = "Scaled_Potentials/O16n_t0t3_L=0.6/u.dat"
    file = "Densities/rho_ca40_t0t3.dat"
    nucl_name = "Ca40n_t0t3"

#loading u functions from C++ code
rC, uC, orbC = loadUF(U_name)

nucl_orig = Problem(Z=part,N=part,max_iter=4000, ub=11, debug='y', basis=ShellModelBasis(),\
               data=quickLoad(file),\
               constr_viol=1e-4, output_folder=nucl_name+"_out", rel_tol=1e-4)

res, info = nucl_orig.solve()
u = res['u']
r = res['grid']

########### SCALED PROBLEM
data = quickLoad(file)
dens = interpolate(data[0], data[1])
t = 1

# loading f functions of the original problem and potentials from C++ code
if n==0:
    F_name = "Results\HO_20_particles_out/f.dat"
    rCC, vCC = quickLoad("Scaled_Potentials/HO_20_particles_L=0.62/Potential", beg=3, end=3)
elif n==1:
    F_name = "Results/"+nucl_name+"_out/f.dat"
    rCC, vCC = quickLoad("Scaled_Potentials\O16n_t0t3_L=0.6/Potential")
elif n==2:
    F_name = "Results/"+nucl_name+"_out/f.dat"
    # rCC, vCC = quickLoad("Scaled_Potentials\Ca40n_t0t3_L=0.6/Potential")
    
    
rho = scaleDensityFun(dens, t, 'l')

cutoff = 1e-9
for rr in np.arange(0.,50.,0.1):
    if( rho(rr) < cutoff ):
        bound = rr - 0.1
        break

print("lambda: ", t)
nucl_scaled = Problem(Z=part, N=part, n_type='p', max_iter=2000, ub=bound, debug='y', \
                  basis=ShellModelBasis(), rho=rho,\
                  exact_hess=True, output_folder=nucl_name+"_out"+str(t),\
                  rel_tol=1e-4, constr_viol=1e-4)

results, info = nucl_scaled.solve()
scaled_u = results['u']
scaled_r = results['grid']


# plotting orbitals, both scaled, theoretical and from IKS inversion
for j in range(u.shape[0]):
    u_fun = interpolate(r, u[j])
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(r, u[j], label=nucl_orig.orbital_set[j].name+r" for $\lambda=1$")
    ax.plot(scaled_r, scaled_u[j], \
            label=nucl_scaled.orbital_set[j].name+r" Python for $\lambda=$"+str(t))
    # ax.plot(rC[j], uC[j], label=nucl_orig.orbital_set[j].name+" C++ for $\lambda=$"+str(t))
    ax.plot(r, t**0.5 * u_fun(t*r), ls='--', label="theoretical scaling")   
    ax.legend(); ax.grid()
    ax.set_title("u(r) for lambda=" + str(t) + " " + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])


## Potential calculations
solver_orig = Solver(nucl_orig)
solver_orig.solve()
r1 = solver_orig.grid
v1 = solver_orig.getPotential()

solver_scaled = Solver(nucl_scaled)
solver_scaled.solve()
r2 = solver_scaled.grid
v2 = solver_scaled.getPotential()

# HF
if n==1:
    out = read("Potentials\pot_o16_t0t3.dat")
    rp, vp = out[0], out[1]
elif n==2:
    out = read("Potentials\pot_ca40_t0t3.dat")
    rp, vp = out[0], out[1]

# computing potential with theoretical scaling
r_orig, f_orig, orb_orig = loadUF(F_name)
f_theor=[]
for i in range(len(f_orig)):
    f_fun = interpolate(r_orig[i], f_orig[i])
    f_theor.append( f_fun(t*r2) )
    
solver_theor = Solver(nucl_scaled, np.array(f_theor).flatten())
x_theor, check = solver_theor.solve()
r_theor = solver_theor.grid
v_theor = solver_theor.getPotential()

"""
# plotting f functions:     
r_scaled, f_scaled, orb_scaled= loadUF("Results/"+nucl_name+"_out0.6/f.dat")
for j in range(f_orig.shape[0]):
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(r_orig[j], f_orig[j], label=nucl_orig.orbital_set[j].name+r" for $\lambda=1$")
    ax.plot(r_scaled[j], f_scaled[j], \
            label=nucl_scaled.orbital_set[j].name+r" Python for $\lambda=$"+str(t))
    ax.plot(r_theor, f_theor[j], ls='--', label="theoretical scaling")   
    ax.legend(); ax.grid()
    ax.set_title("f(r) for lambda=" + str(t) + " " + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel("f(r)")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])
"""

if n==0:
    shift = 3
else: 
    shift = -20

# plotting potentials 
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(r1, v1 - v1[shift], label=r"Python")
if n==0:
    r_HO = np.arange(0.,15.,0.1)
    v_HO = 0.5*m_p*(nuclearOmega(part))**2*r_HO**2
    ax.plot(r_HO, v_HO - v_HO[3], ls='--', label= "HO Theorical")
    
    v_HO_lam_mod = 0.5*m_p*(t**2*nuclearOmega(part))**2*r_HO**2
else:
    ax.plot(rp, vp - vp[shift], ls='--', label= "HF")
ax.legend(); ax.grid()
ax.set_title(r"Potentials with $\lambda = 1$")
ax.set_xlabel("r")
ax.set_ylabel("v")
if n==0:
    ax.set_xlim([0, 11])
    ax.set_ylim([0, 350])
else:
    ax.set_xlim([0, 10.5])
    ax.set_ylim([-80, 5])


fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(r1, v1 - v1[shift], label=r"Python L=1")
ax.plot(r2, v2 - v2[shift], label=r"Python")
if n!=2: # C code
    ax.plot(rCC, vCC - vCC[shift], ls='--', label=r"C++")
ax.plot(r_theor, v_theor - v_theor[shift], ls='--', label= "Theoretical")
if n==0: # HO
    ax.plot(r_HO, v_HO - v_HO[3], ls='--', label= "Theorical not scaled", color="yellow")
    ax.plot(r_HO, v_HO_lam_mod - v_HO_lam_mod[shift], ls='--', label= r"Applying scaling on $\nu$")
ax.legend(); ax.grid()
ax.set_title(r"Potentials with $\lambda = $ "+str(t))
ax.set_xlabel("r")
ax.set_ylabel("v")
if n==0:
    ax.set_xlim([0, 10.9])
    ax.set_ylim([0, 55])
else:
    ax.set_xlim([0, 10.5])
    ax.set_ylim([-80, 5])


#%%

new_u = solver_orig.getDiagOrbitals()
diag_orig_u = np.zeros_like(new_u)
if part==20:
    for j in range(new_u.shape[0]):
        if solver_scaled.sorted_orbital_set[j].name=="2s 1/2" or \
            solver_scaled.sorted_orbital_set[j].name=="1s 1/2":
            diag_orig_u[j] = -new_u[j]
        else:
            diag_orig_u[j] = new_u[j]
else:
    diag_orig_u = new_u

new_u = solver_scaled.getDiagOrbitals()
diag_u = np.zeros_like(new_u)
if part==20:
    for j in range(new_u.shape[0]):
        if solver_scaled.sorted_orbital_set[j].name=="2s 1/2" or \
            solver_scaled.sorted_orbital_set[j].name=="1s 1/2":
            diag_u[j] = -new_u[j]
        else:
            diag_u[j] = new_u[j]
else:
    diag_u = new_u
    
# """   
for j in range(u.shape[0]):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    if solver_scaled.sorted_orbital_set[j].name=="2s 1/2":
        j = j-1
    elif solver_scaled.sorted_orbital_set[j].name=="1d 5/2":
        j = j+1
    ax.plot(r, diag_orig_u[j], label=solver_scaled.sorted_orbital_set[j].name+" post diag orig")
    ax.plot(r2, diag_u[j], ls='--', label=solver_scaled.sorted_orbital_set[j].name+" post diag scaled")
    ax.legend(); ax.grid()
    ax.set_title("u(r) for lambda=" + str(t) + " " + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
"""

# plotting orbitals, both scaled, theoretical and from IKS inversion
for j in range(u.shape[0]):
    u_fun = interpolate(r, u[j])
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(r, u[j], label=nucl_orig.orbital_set[j].name+r" for $\lambda=1$")
    ax.plot(scaled_r, scaled_u[j], \
            label=nucl_scaled.orbital_set[j].name+r" Python for $\lambda=$"+str(t))
    # ax.plot(rC[j], uC[j], label=nucl_orig.orbital_set[j].name+" C++ for $\lambda=$"+str(t))
    ax.plot(r, t**0.5 * u_fun(t*r), ls='--', label="theoretical scaling")   
    if solver_scaled.sorted_orbital_set[j].name=="2s 1/2":
        j = j-1
    elif solver_scaled.sorted_orbital_set[j].name=="1d 5/2":
        j = j+1
    ax.plot(r2, diag_u[j], label=solver_scaled.sorted_orbital_set[j].name+"post diag")
    ax.legend(); ax.grid()
    ax.set_title("u(r) for lambda=" + str(t) + " " + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])
# """

#%% HF u FUNCTIONS COMPARISON
r_HF=[]; u_HF=[]
if n==1:
    r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1s12.dat")
    r_HF.append(r_temp); u_HF.append(u_temp)
    r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1p32.dat")
    r_HF.append(r_temp); u_HF.append(u_temp)
    r_temp, u_temp = quickLoad("HF_orbitals/O16n_t0t3_1p12.dat")
    r_HF.append(r_temp); u_HF.append(u_temp)
elif n==2:
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

# plotting orbitals, both scaled, theoretical and from IKS inversion
for j in range(u.shape[0]):
    u_fun = interpolate(r, u[j])
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(r, u[j], label=nucl_orig.orbital_set[j].name+r" IKS Python")
    ax.plot(r_HF[j], u_HF[j], ls='--', label=nucl_orig.orbital_set[j].name+" HF" )
    if solver_scaled.sorted_orbital_set[j].name=="2s 1/2":
        j = j-1
    elif solver_scaled.sorted_orbital_set[j].name=="1d 5/2":
        j = j+1
    ax.plot(r2, diag_u[j], label=nucl_orig.orbital_set[j].name+r" IKS Python post diag")
    ax.legend(); ax.grid()
    ax.set_title(r"u(r) for $\lambda=1$ " + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])
    
#%% THEORETICAL HO 3D
from EigenFunctions import HO_3D
nu = nuclearNu(20)
ho3d=[]
ho3d.append(HO_3D(1,0,nu)) # 1s 1/2
ho3d.append(HO_3D(1,1,nu)) # 1p 3/2
ho3d.append(HO_3D(1,1,nu)) # 1p 1/2
ho3d.append(HO_3D(1,2,nu)) # 1d 5/2
ho3d.append(HO_3D(2,0,nu)) # 2s 1/2
ho3d.append(HO_3D(1,2,nu)) # 1d 3/2

# plotting orbitals, both scaled, theoretical and from IKS inversion
for j in range(u.shape[0]):
    u_fun = interpolate(r, u[j])
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    # ax.plot(r, u[j], label=nucl_orig.orbital_set[j].name+r" for $\lambda=1$")
    # ax.plot(scaled_r, scaled_u[j], \
            # label=nucl_scaled.orbital_set[j].name+r" Python for $\lambda=$"+str(t))
    # ax.plot(rC[j], uC[j], label=nucl_orig.orbital_set[j].name+" C++ for $\lambda=$"+str(t))
    # ax.plot(r, t**0.5 * u_fun(t*r), label=nucl_orig.orbital_set[j].name+" u(r) from IKS")   
    ax.plot(r, t**1.5 * ho3d[j](t*r)*r, label=nucl_orig.orbital_set[j].name+" analytical R(r)")
    if solver_scaled.sorted_orbital_set[j].name=="2s 1/2":
        j = j-1
    elif solver_scaled.sorted_orbital_set[j].name=="1d 5/2":
        j = j+1
    ax.plot(r2, diag_u[j], ls='--', label=solver_scaled.sorted_orbital_set[j].name+"post diag")
    ax.legend(); ax.grid()
    ax.set_title("u(r) for lambda=" + str(t) + " " + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 20])
    
    """
    ax.plot(r, u[j], label=nucl_orig.orbital_set[j].name+r" for $\lambda=1$")
    ax.plot(r, ho3d[j](r)*r, ls='--', label=nucl_orig.orbital_set[j].name+" analytical R(r)") 
    """
    
   
#%% PLOTTING THEORETICAL DIFFERENCES
diff = []
if n==0:
    for j in range(u.shape[0]):
        if solver_scaled.sorted_orbital_set[j].name=="2s 1/2":
            x = j-1
        elif solver_scaled.sorted_orbital_set[j].name=="1d 5/2":
            x = j+1
        else:
            x=j
        diff.append( ho3d[j](r2)*r2 - diag_u[x] )
    R=r2
else:
    for j in range(u.shape[0]):
        if solver_scaled.sorted_orbital_set[j].name=="2s 1/2":
            x = j-1
        elif solver_scaled.sorted_orbital_set[j].name=="1d 5/2":
            x = j+1
        else:
            x=j
        diag_u_fun = interpolate(r2, diag_u[x])
        diff.append( u_HF[j] - diag_u_fun(r_HF[j]) )
    R=r_HF[j]
    
for j in range(u.shape[0]):    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    
    ax.plot(R, diff[j], label=solver_scaled.sorted_orbital_set[j].name+" Theor - IKS")
    ax.legend(); ax.grid()
    ax.set_title(r"$\Delta$u(r) for $\lambda$=1 " + nucl_name)
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    ax.set_xlim([0, 10])
    if n!=0:
        ax.set_ylim([-0.002, 0.002])  
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# %%
## Potential energy calc
v1 = v1 - v1[-10]
elim = np.arange(-50,0,1)
v = np.delete(v1, elim)
r = np.delete(r1, elim)
U1 = integrate.simpson(4*np.pi*r**2*dens(r)*v, r)
print(U1)

# HF ??
U2 = integrate.simpson(4*np.pi*rp**2*dens(rp)*vp, rp)
print(U2)
