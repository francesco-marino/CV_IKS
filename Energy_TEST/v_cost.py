# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 17:45:14 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

from Energy import Energy, quickLoad

def plotEnergy(E1, r1, scaling):
        
        assert ( len(E1)==len(r1) ) 
        
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(
            r1, E1,  
            label = scaling + " Energy",
            color = "orange",
            lw = 2
            )
        plt.grid(); plt.legend()
        ax.set_title(scaling + " scaling")
        ax.set_xlabel("Volume radius"),
        ax.set_ylabel("Energy")
        # ax.set_ylim([-0.01, 0.01])
        
if __name__ == "__main__":
    
    lower_R = 5.
    upper_R = 60.
    a = 3.
    
    #defining theoretical energy 
    # E_theor = lambda r, rho : 4*np.pi * integrate.simpson(a * r**2 * rho(r)**2,r)
    # rho_norm = lambda r: rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)
    
    #defining potential
    def potential(r, rho, N=1, t=1, rr=[]):

        return 3.*np.ones_like(r)
    
    name = "Densities\SkXDensityO16p.dat"
    
    print("\n \n TEST 1: ------------ v = constant \n \n")
    #TEST 1
    
    data = quickLoad(name)
    
    EQ = []; EL = []; EZ = []
    R = np.arange(lower_R, upper_R, 1)
    for i in R:
        energy = Energy(data=data, v=potential, \
                        param_step=0.001, r_step=0.01, R_min=0., R_max=i,\
                        scaling='l', t_min=0.2)
        E = energy.solver()
        EL.append(E)
        print(i)
    
        
    plotEnergy(EL, R, 'L')
