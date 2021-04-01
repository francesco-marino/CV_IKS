# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:56:47 2021

@author: alberto
"""

import numpy as np
from Orbitals import ShellModelBasis
from Problem import Problem
from Energy_2_0 import Energy

"""
TEST useful just to check whether the integral implementation is correct. 
Energies obtained are meaningless.

In this case: rho = r, v = rho**2
"""

class Energy_dummy(Energy):
    def __init__(self, problem, output="Output", param_step=0.1, r_step=0.1, R_min=0.01, R_max=10., potential=[]):

        super().__init__(problem, output, param_step, r_step, R_min, R_max)
        
        self.potential = potential
        
    def param_Potential(self):
        r = self.potential[0]
        p = self.potential[1]
        
        v=[]
        for t in self.T:
            v.append(t**2*p) #HERE is the difference between test 1 and 2
            
        t_col=np.reshape(self.T,newshape=(-1,1))
        #print("\n t_col \t", np.shape(t_col))
        
        return v, v/(3*t_col), v/(2*np.sqrt(t_col)), r
    
    def solver(self):
        
        # computing potentials with IKS
        self.vQ, self.vL, self.vZ, self.v_grid = self.param_Potential()
        
        #density for given r of the integral
        rho_int = self.rho_fun(self.R)
        Drho_int= self.d_dx(rho_int)
        #potential for given r of the integral
        v_int_Q = self._evalPotential(self.vQ)
        v_int_L = self._evalPotential(self.vL)
        v_int_Z =  self._evalPotential(self.vZ)
        #print("v Q",v_int_Q, "\n\n L", v_int_L, "\n\n Drho", Drho_int, "\n\n Z", v_int_Z)
        
        E = self.calcIntegral(rho_int, Drho_int, v_int_Q, v_int_L, v_int_Z)
        
        return E
    
##########TEST########

r = np.array([r for r in np.arange(0,15,0.1)])
rho = r
potential = rho**2

problem = Problem(Z=2, data=(r,rho), basis=ShellModelBasis())

energy = Energy_dummy(problem, potential=(r,potential), param_step=0.01, r_step=0.01, R_max=5.)
E = energy.solver()
print(E)

#EXPECTED (THEORETICAL) VALUES:
# 13089 - 21816 - 15271

#VALUES FROM SIMULATION:
#param_step = 0.01: 14226 - 19999 - 16560
