# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 23:33:54 2021

@author: alberto
"""

import numpy as np

from Energy import Energy


if __name__ == "__main__":
    
    # problem_IKS = Problem(Z=20,N=20, max_iter=10, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SkXDensityCa40p.dat"))
    #problem_IKS = Problem(Z=20,N=20, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat"))
    #problem_IKS = Problem(Z=82,N=126, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SOGDensityPb208p.dat"))
    
    # energy = Energy(problem_IKS, "Ca40SkX_En")
    #energy = Energy(problem_IKS, "HO20coupled")
    #energy = Energy(problem_IKS, "Pb208SOG_En")
    
    print("\n \n TEST 1: ------------ \n \n")
    #TEST 1
    rho = lambda r : np.ones_like(r)
    grad_rho = lambda r : np.zeros_like(r)
    # potential = lambda rho : rho
    def potential(r, rho, a=1):
        v = a * rho(r)

        return v
    
    # energy = Energy(rho=rho, v=potential, grad_rho=grad_rho, param_step=0.001, r_step=0.001, R_min=0., R_max=10.)
    energy = Energy(rho=rho, v=potential, param_step=0.001, r_step=0.001, R_min=0., R_max=10.)
    
    E = energy.solver()
    print("Energies: \n",\
          "\n\nEnergies with trapezoids: ", E[0],\
          "\n\nEnergies from simpson: ", E[1])
        

    print("\n \n TEST 2: ------------ \n \n")
    #TEST 2   
    rho = lambda r : r
    grad_rho = lambda r : np.ones_like(r)
    # potential = lambda rho : rho**2
    # potential = lambda rho,r : rho(r)**2
    
    def potential(r, rho, a=1):
        v = (a * rho(r)) **2

        return v
        
    # energy = Energy(rho=rho, v=potential, grad_rho=grad_rho, param_step=0.001, r_step=0.01, R_min=0., R_max=5.)
    energy = Energy(rho=rho, v=potential, param_step=0.001, r_step=0.001, R_min=0., R_max=5., output="prova")
    
    
    E = energy.solver()
    print("Energies: \n",\
          "\n\nEnergies with trapezoids: ", E[0],\
          "\n\nEnergies from simpson: ", E[1])