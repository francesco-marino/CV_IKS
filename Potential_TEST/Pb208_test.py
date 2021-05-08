# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 00:13:41 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Misc import loadData, read
from Problem import Problem, quickLoad
from Problem_mod import Problem_mod
from Problem_mod2 import Problem_mod2
from Solver import Solver
from Orbitals import ShellModelBasis

if __name__ == "__main__":
    for bound in [10., 11., 12., 13.]:
        print("using NO MOD \n")
        nucl0 = Problem(Z=82,N=126, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=quickLoad("Densities/SOGDensityPb208p.dat"), \
                       output_folder="Pb208SOG", exact_hess=True)
        
        data, info = nucl0.solve()
        data = loadData(nucl0.datafile)
        status = data['status']
        x = data['x']
    
        solver0 = Solver(nucl0, x)
        x, check = solver0.solve()
        print (check)
        
        pot0 = solver0.getPotential()
        
        print("using MOD \n")
        nucl1 = Problem_mod(Z=82,N=126, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=quickLoad("Densities/SOGDensityPb208p.dat"), \
                       output_folder="Pb208SOG", exact_hess=True)
            
        data, info = nucl1.solve()
        data = loadData(nucl1.datafile)
        status = data['status']
        x = data['x']
    
        solver1 = Solver(nucl1, x)
        x, check = solver1.solve()
        print (check)
        
        pot1 = solver1.getPotential()
        
        print("using MOD 2 \n")
        nucl2 = Problem_mod2(Z=82,N=126, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=quickLoad("Densities/SOGDensityPb208p.dat"), \
                       output_folder="Pb208SOG", exact_hess=True)
            
        data, info = nucl2.solve()
        data = loadData(nucl2.datafile)
        status = data['status']
        x = data['x']
    
        solver2 = Solver(nucl2, x)
        x, check = solver2.solve()
        print (check)
        
        pot2 = solver2.getPotential()
        
        #Benchmark 
        r = np.arange(0, 10, 0.1)
        WS = lambda r : -52*(1+np.exp((r-7.4)/0.7))**-1 #Woods Saxon
        
        out = read("Potentials\pot_pb208_SOG_other_iks.dat")
        r_other, vp_other = out[0], out[1]
        
        #NO MOD:: ub=8, ub=9 DO NOT converge, ub=10 restoration failed (still got a potential tho), ub=11 iterations exceeded
        #MOD 2:: ub=9 infeasibility, ub=10 maax iter exceeded (still got a potential tho)
        
        #plotting results
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(solver0.grid, pot0 - pot0[65]+WS(6.5), label="acc=2")
        ax.plot(solver1.grid, pot1 - pot1[65]+WS(6.5), label="acc=6")
        ax.plot(solver2.grid, pot2 - pot2[65]+WS(6.5), label="acc=4")
        ax.plot(r, WS(r), '--', label="exact")
        ax.plot(r_other, vp_other - vp_other[62]+WS(6.5), label="other IKS")
        plt.grid(); plt.legend()
        ax.set_title("Pb208SOG " +str(bound) + " mod")
        ax.set_xlabel("radius")
        ax.set_ylabel("potential")
        ax.set_xlim([0, 10])
        ax.set_ylim([WS(3)-20, 20])