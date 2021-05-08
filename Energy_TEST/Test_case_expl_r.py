# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:49:10 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

from Energy import Energy
from Energy_old import Energy_old

def plotEnergy(E1, r1, E2, r2, scaling, rho_type, output="Case_Expl(r)", save='y'):
        
        assert ( len(E1)==len(r1) ) and ( len(E2)==len(r2) )
        
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(
            r1, E1,  
            label = scaling + " Energy",
            color = "orange",
            lw = 2
            )
        ax.plot(
            r2, E2,  
            '--', 
            label = "Theoretical Energy",
            color = "blue",
            lw = 2
            )
        plt.grid(); plt.legend()
        ax.set_title(scaling + " scaling")
        ax.set_xlabel("Volume radius"),
        ax.set_ylabel("Energy")
        
        if(save == 'y'):
            if not os.path.exists("Energy_TEST/" + output):
                os.makedirs("Energy_TEST/" + output)
                
            out_fold = "Energy_TEST/" + output
            file = "/test_case1_" + scaling + "_rho=" + rho_type + ".png"
            plt.savefig(out_fold + file)

if __name__ == "__main__":
    
    lower_R = 5.
    upper_R = 20.

    rho = None
    
    #defining theoretical energy 
    E_theor = lambda r, rho : 4*np.pi * integrate.simpson(r**2 * ( rho(r)**2 + rho(r)*r**2 ), r)
    rho_norm = lambda r: rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)
    
    #defining potential
    #potential with explicit dependence on r (not only rho!)
    def potential(r, rho, N=1, t=1, rr=[]):
        if not len(rr)>0 : rr=r
        v = 2 * t * rho(r) / N + rr**2

        return v
    
    
    print("\n \n TEST 1: ------------ Rho = fermi function \n \n")
    #TEST 1
    
    a = 50.
    aa = 0.5
    RR = 10.

    rho = lambda r : a * (1+np.exp((r-RR)/aa))**-1
    grad_rho = lambda r : (-1) * a/aa * np.exp((r-RR)/aa) * (1+np.exp((r-RR)/aa))**-2
    
    EQ = []; EL = []; EZ = []
    EQ1 = []; EL1 = []; EZ1 = []
    R = np.arange(lower_R, upper_R, 0.1)
    # R=[20]
    for i in R:
        # """
        energy = Energy(rho=rho, v=potential, scaling='all', \
                        param_step=0.001, r_step=0.001, R_min=0., R_max=i)
        """
        energy_old = Energy_old(rho=rho, v=potential, scaling='all', \
                        param_step=0.001, r_step=0.001, R_min=0., R_max=i)
        """
        E = energy.solver()
        EQ.append(-E[0])
        EL.append(-E[1])
        EZ.append(-E[2])
        """
        E = energy_old.solver()
        EQ1.append(E[0])
        EL1.append(E[1])
        EZ1.append(E[2])
        # """       
        
    # """
    E_th = []
    for i in R:
        r = np.arange(0.001, i, 0.001)
        e = E_theor(r, rho_norm)
        E_th.append(e)

    plotEnergy(EQ, R, E_th, R, 'Q', "EXP", save='n')
    plotEnergy(EL, R, E_th, R, 'L', "EXP", save='n')
    plotEnergy(EZ, R, E_th, R, 'Z', "EXP", save='n')
    # plotEnergy(EQ1, R, E_th, R, 'Q', "EXP", save='n')
    # plotEnergy(EL1, R, E_th, R, 'L', "EXP", save='n')
    # plotEnergy(EZ1, R, E_th, R, 'Z', "EXP", save='n')
    # """
    
# %%
R_max=20.; R_min=0.; step=0.01
num = int((R_max - R_min) / step)
R = np.linspace(R_min, R_max, num)
print(R, len(R))
R = np.arange(R_min, R_max + step, step)
print(R, len(R))
