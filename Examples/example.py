"""
A complete working example

"""

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from Problem  import Problem, quickLoad
from Solver import Solver
from Misc import loadData, read
from Orbitals import ShellModelBasis



if __name__=="__main__":
    
    """
    "quickLoad" is a practical routine to load the first two columns of 
    a file, interpreted as the postions (r) and densities (rho). 
    """
    data = quickLoad("../Densities/rho_ca40_t0t3.dat")
    
    
    """
    Setting up an IKS problem.
    Using a proton density for the O16 nucleus.
    """
    nucl = Problem(Z=20,N=20, n_type='p', h=0.1, max_iter=4000, \
        ub=11, basis=ShellModelBasis(), data=data, \
        output_folder="17_giugno", com_correction=True )
    
    
    """
    Perform the minimization.
    Returns a dictionary of important results and a complete report 
    of ipopt (info)
    """
    results, info = nucl.solve()
    
    """
    A Solver object is created and take Problem as its arguments.
    Solver shall be used to determine the potential and the eigenvalues.
    """
    solver = Solver(nucl)
    
    """
    Pot. and eigenvalues are shifted automatically if cost=None, but one can pass
    a user-defined offset to be subtracted.
    """
    solver.solve(shift=True, cost=None)
    
   
    # Plot the potential
    plt.figure(0)
    plt.plot(solver.grid, solver.potential , '--', label="CV")
    
    out = read("../Potentials/pot_ca40_t0t3.dat")
    r, vp = out[0], out[1]
    plt.plot(r, vp, '--', label="HF")
    
    plt.xlim(0.,10.); plt.ylim(-70.,25.)
    plt.grid(); plt.legend()
    
    # Print eigenvalues
    print ("Eigenvalues")
    for j in range(solver.n_orbitals):
        print ( str(solver.orbital_set[j]), "\t", solver.eigenvalues[j] )
    
    
    # Plot the eigenvectors
    plt.figure(1)
    for j in range(solver.n_orbitals):
        plt.plot(solver.grid, solver.eigenvectors[j,:], ls='--', label=solver.orbital_set[j].name)
    plt.legend()
    
    
    
    
    
    