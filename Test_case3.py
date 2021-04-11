# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:58:23 2021

@author: alberto
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sns

from Energy import Energy

"""
Plotting energies a*rho**(b+1) varying volume radius and b.
"""

def calc(rho, potential):
    EQ = []; EL = []; EZ = []
    R = np.arange(lower_R, upper_R, 0.1)
    for i in R:
        energy = Energy(rho=rho, v=potential, \
                        param_step=0.001, r_step=0.01, R_min=0., R_max=i)
        E = energy.solver()
        EQ.append(E[0])
        EL.append(E[1])
        EZ.append(E[2])
    
    E_th = []
    for i in R:
        r = np.arange(0.001, i, 0.001)
        e = E_theor(r, rho_norm)
        E_th.append(e)
    return EQ, EL, EZ, E_th, R


def plot(R, E, ax):
           
    g = sns.lineplot(
        x=R, y=np.log10(E),
        ax=ax,
        label = "b=" + str(b),
        )
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
     

a = 5.
lower_R = 5.
upper_R = 20.

c = 50.
aa = 0.5
RR = 10.

x = np.arange(5.,20.01,1.)
sns.set_theme(style="white", palette='nipy_spectral', font_scale = 1.5)

fig,ax1=plt.subplots(figsize=(10,6))
ax1.set_xlabel("Volume radius")
ax1.set_ylabel("Energies [logscale]")
ax1.set_title("Constant")
ax1.set_xticks(x)


fig,ax2=plt.subplots(figsize=(10,6))
ax2.set_xlabel("Volume radius")
ax2.set_ylabel("Energies [logscale]")
ax2.set_title("Fermi")
ax2.set_xticks(x)


for b in range(0,10):
    print("Simulation with b = " + str('%.0f'%b))
    E_theor = lambda r, rho : 4*np.pi * integrate.simpson(a * r**2 * rho(r)**(b+1.),r)
    
    def potential(r, rho, N=1, t=1):
        v = a * (b+1.) * (t*rho(r)/N)**b

        return v
    
    #costant
    rho = lambda r : a * np.ones_like(r)
    rho_norm = lambda r : rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)
    
    E1,E2,E3,E4 , R= calc(rho, potential)
    """
    E_const=[]
    for E in [E1,E2,E3,E4]:
        E_const.append(E)
    E_const = np.array(E_const)
    """        
    plot(R, E1, ax1)
    
    #fermi
    rho = lambda r : c * (1+np.exp((r-RR)/aa))**-1
    rho_norm = lambda r : rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)
    
    E1,E2,E3,E4 , R= calc(rho, potential)
    """
    E_fermi=[]
    for E in [E1,E2,E3,E4]:
        E_fermi.append(E)
    E_fermi=np.array(E_fermi)
    """
    plot(R, E1, ax2)