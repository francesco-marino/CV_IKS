# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:38:29 2021

@author: alberto
"""

import numpy as np

from Problem import Problem, quickLoad
from Solver import Solver
from Orbitals import ShellModelBasis
from Energy import Energy 


##########------------SOLVER & PROBLEM TEST--------############

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
#plt.plot(s0.grid, s0.getPotential(), ls='-.', c='r',  label="Init")
plt.grid(); plt.legend()
ax.set_title(file)
ax.set_xlabel("radius"),
ax.set_ylabel("potential")
ax.set_xlim([0, 9.7])
ax.set_ylim([-100, 10])

#%%
##########------------ENERGY TEST--------############

#energy = Energy(Z=20,N=0, max_iter=10, rel_tol=1e-4, constr_viol=1e-4, param_step=0.1, r_step=0.1, file="Densities/rho_HO_20_particles_coupled_basis.dat", output="HO20coupled")
energy = Energy(Z=20,N=20, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, param_step=0.1, r_step=0.1, file="Densities/SkXDensityCa40p.dat", output="Ca40SkX_En")
#energy = Energy(Z=82,N=126, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, param_step=0.1, r_step=0.1, file="Densities/SOGDensityPb208p.dat", output="Pb208SOG_En")
E = energy.solver()
print("Energies", E)