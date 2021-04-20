# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:37:28 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Misc import loadData
from Orbitals import getOrbitalSet, ShellModelBasis
from Problem import Problem, getSampleDensity, quickLoad
from Solver import Solver

if __name__=="__main__":
    for n in [8, 20, 28, 50, 82, 126]: 
        Orbs = getOrbitalSet(n_particles=n, basis=ShellModelBasis())
        
        HO_rho = getSampleDensity(len(Orbs))
        
        r = np.arange(0.,20.,0.1)
        
        data = np.column_stack((r, HO_rho(r)))
        np.savetxt("Densities/rho_HO_"+str(n)+"_particles_coupled_basis.dat", data)
        
        data = r, HO_rho(r)
        problem = Problem(Z=n,n_type='p', max_iter=4000, ub=12., debug='y', \
                              basis=ShellModelBasis(), data=quickLoad("Densities/rho_HO_"+str(n)+"_particles_coupled_basis.dat"),\
                              exact_hess=True) 
        res, info = problem.solve()
        res = loadData(problem.output_folder+"\data")
        x = res['x']
        
        solver = Solver(problem)
        x, check = solver.solve()
        print (check)
        
        pot = solver.getPotential()
        
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(
            solver.grid, pot,  
            color = "orange",
            label = "CV", 
            lw = 2
            )
        ax.plot(
            solver.grid, solver.grid**2 + pot[0],  
            color = "blue",
            label = "quadratic", 
            lw = 2
            )
        plt.grid(); plt.legend()
        ax.set_title("HO potential with n="+str(n))
        ax.set_xlabel("Radius r")
        ax.set_xlim([0, 15])
        ax.set_ylabel("Potential v")

        fig2, ax = plt.subplots(1,1,figsize=(5,5))
        u = problem.results['u']
        for j in range(u.shape[0]):
            ax.plot(problem.grid, problem.getU(problem.results['start'])[j,:], label=problem.orbital_set[j].name+"  INIT")
            ax.plot(problem.grid, u[j,:], ls='--', label=problem.orbital_set[j].name)
        ax.legend()
