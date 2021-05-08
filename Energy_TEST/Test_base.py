# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:38:29 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Misc import loadData
from Problem import Problem, quickLoad
from Solver import Solver
from Orbitals import ShellModelBasis
from Energy import Energy 


##########------------SOLVER & PROBLEM TEST--------############

if __name__ == "__main__":

    #file = "Densities/rho_HO_20_particles_coupled_basis.dat"
    
    #nucl = Problem( OrbitalSet(basis[:n_orb]).countParticles(), rho=rho,basis=basis)
    nucl = Problem(Z=82,N=126,max_iter=4000, ub=10., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/SOGDensityPb208p.dat") )
    #nucl = Problem(Z=20,N=20,max_iter=4000, ub=10., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityCa40p.dat") )
    
    data, info = nucl.solve()
    data = loadData(nucl.datafile)
    #x0 = nucl.getStartingPoint()
    status = data['status']
    x = data['x']
    
    #s0 = Solver(nucl, x0)
    solver = Solver(nucl, x)
    x, check = solver.solve()
    print (check)
    
    
    #plotting results
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(solver.grid, solver.getPotential(), '--', label="Solution")
    plt.grid(); plt.legend()
    ax.set_title("Potential")
    ax.set_xlabel("radius"),
    ax.set_ylabel("potential")
    ax.set_xlim([0, 9.7])
    ax.set_ylim([-100, 10])

#%%
##########------------IKS ENERGY TEST--------############

if __name__ == "__main__":

    problem_IKS = Problem(Z=20,N=20, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SkXDensityCa40p.dat"))
    #problem_IKS = Problem(Z=20,N=20, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat"))
    #problem_IKS = Problem(Z=82,N=126, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SOGDensityPb208p.dat"))
    
    energy = Energy(problem_IKS, "Ca40SkX_En")
    #energy = Energy(problem_IKS, "HO20coupled")
    #energy = Energy(problem_IKS, "Pb208SOG_En")
    
    E = energy.solver()
    print("Energies", E)
