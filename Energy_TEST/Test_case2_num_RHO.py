# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:09:48 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

from Problem import quickLoad
from Energy import Energy

def plotEnergy(E1, r1, E2, r2, scaling, rho_type, output="Case2", save='y'):
        
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
        ax.set_ylim([-3, 5])
        
        if(save == 'y'):
            if not os.path.exists("Energy_TEST/" + output + "_num_RHO"):
                os.makedirs("Energy_TEST/" + output + "_num_RHO")
                
            out_fold = "Energy_TEST/" + output + "_num_RHO"
            file = "/Num_RHO_test_case2_" + scaling + "_rho=" + rho_type + ".png"
            plt.savefig(out_fold + file)

if __name__ == "__main__":
    
    lower_R = 5.
    upper_R = 10.
    a = 3.
    
    #defining theoretical energy 
    E_theor = lambda r, rho : 4*np.pi * integrate.simpson(r**2 * rho(r),r)
    rho_norm = lambda r : rho(r) / integrate.simpson(rho(r) * r**2 *4*np.pi, r)
    
    #defining potential
    def potential(r, rho, N=1, t=1):
        v = np.ones_like(r)

        return v
    
    
    print("\n \n TEST 1: ------------ Rho = constant \n \n")
    #TEST 1
    
    rho = lambda r : a * np.ones_like(r)
    grad_rho = lambda r : np.zeros_like(r)
    
    EQ = []; EL = []; EZ = []
    R = np.arange(lower_R, upper_R, 0.1)
    
    r = np.arange(0.,20.,0.1)
    arr_rho = rho(r)
    
    data = np.column_stack((r, arr_rho))
    np.savetxt("Densities/prova_rho.dat", data)
    
    E_th = []
    for i in R:
        energy = Energy(rho=None, v=potential, grad_rho=None, data=quickLoad("Densities/prova_rho.dat"),\
                        param_step=0.001, r_step=0.01, R_min=0.,\
                        R_max=i) #, ref_rho=ref_rho)
        E = energy.solver()
        EQ.append(E[0])
        EL.append(E[1])
        EZ.append(E[2])
        
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
    
    r = np.arange(0.,20.,0.1)
    arr_rho = rho(r)
    
    data = np.column_stack((r, arr_rho))
    np.savetxt("Densities/prova_rho.dat", data)
    
    E_th = []
    for i in R:
        energy = Energy(rho=None, v=potential, grad_rho=None, data=quickLoad("Densities/prova_rho.dat"),\
                        param_step=0.001, r_step=0.01, R_min=0.,\
                        R_max=i) #, ref_rho=ref_rho)
        E = energy.solver()
        EQ.append(E[0])
        EL.append(E[1])
        EZ.append(E[2])
        
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
    
    r = np.arange(0.,20.,0.1)
    arr_rho = rho(r)
    
    data = np.column_stack((r, arr_rho))
    np.savetxt("Densities/prova_rho.dat", data)
    
    E_th = []
    for i in R:
        energy = Energy(rho=None, v=potential, grad_rho=None, data=quickLoad("Densities/prova_rho.dat"),\
                        param_step=0.001, r_step=0.01, R_min=0.,\
                        R_max=i) #, ref_rho=ref_rho)
        E = energy.solver()
        EQ.append(E[0])
        EL.append(E[1])
        EZ.append(E[2])
        
        r = np.arange(0.001, i, 0.001)
        e = E_theor(r, rho_norm)
        E_th.append(e)
   
    plotEnergy(EQ, R, E_th, R, 'Q', "EXP", save='y')
    plotEnergy(EL, R, E_th, R, 'L', "EXP", save='y')
    plotEnergy(EZ, R, E_th, R, 'Z', "EXP", save='y')
