# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 17:45:14 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

from Energy import Energy

def plotEnergy(E1, r1, E2, r2, scaling, rho_type, output="Case1", save='y'):
        
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
    a = 3.
    
    #defining theoretical energy 
    E_theor = lambda r, rho : 4*np.pi * integrate.simpson(a * r**2 * rho(r)**2,r)
    rho_norm = lambda r: rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)
    
    #defining potential
    def potential(r, rho, N=1, t=1, rr=[]):
        if not len(rr)>0 : rr=r
        v = 2*a * t * rho(r) / N

        return v
    
    
    print("\n \n TEST 1: ------------ Rho = constant \n \n")
    #TEST 1
    
    rho = lambda r : a * np.ones_like(r)
    grad_rho = lambda r : np.zeros_like(r)
    
    EQ = []; EL = []; EZ = []
    R = np.arange(lower_R, upper_R, 0.1)
    for i in R:
        energy = Energy(rho=rho, v=potential, grad_rho=grad_rho, scaling='all',\
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
        
    plotEnergy(EQ, R, E_th, R, 'Q', "CONSTANT", save='y')
    plotEnergy(EL, R, E_th, R, 'L', "CONSTANT", save='y')
    plotEnergy(EZ, R, E_th, R, 'Z', "CONSTANT", save='y')
    
    
    print("\n \n TEST 2: ------------ Rho = sin^2(r) \n \n")
    #TEST 2  

    rho = lambda r : np.sin(r)**2
    grad_rho = lambda r : np.sin(2*r)
    
    EQ = []; EL = []; EZ = []
    R = np.arange(lower_R, upper_R, 0.1)
    for i in R:#grad_rho=grad_rho,\
        energy = Energy(rho=rho, v=potential, scaling='all',\
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
        
    plotEnergy(EQ, R, E_th, R, 'Q', "SIN^2", save='y')
    plotEnergy(EL, R, E_th, R, 'L', "SIN^2", save='y')
    plotEnergy(EZ, R, E_th, R, 'Z', "SIN^2", save='y')
    
    
    print("\n \n TEST 3: ------------ Rho = const * (1+exp((r-R)/a))**-1 \n \n")
    #TEST 3
    a = 50.
    aa = 0.5
    RR = 10.

    rho = lambda r : a * (1+np.exp((r-RR)/aa))**-1
    grad_rho = lambda r : (-1) * a/aa * np.exp((r-RR)/aa) * (1+np.exp((r-RR)/aa))**-2
    
    EQ = []; EL = []; EZ = []
    R = np.arange(lower_R, upper_R, 0.1)
    for i in R:#grad_rho=grad_rho,\
        energy = Energy(rho=rho, v=potential, scaling='all', \
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
   
    plotEnergy(EQ, R, E_th, R, 'Q', "EXP", save='y')
    plotEnergy(EL, R, E_th, R, 'L', "EXP", save='y')
    plotEnergy(EZ, R, E_th, R, 'Z', "EXP", save='y')