# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 00:13:41 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Misc import loadData, read
from Problem import Problem, quickLoad
# from Problem_mod import Problem_mod, quickLoad
# from Problem_mod2 import Problem_mod2
from Solver import Solver
from Orbitals import ShellModelBasis

if __name__ == "__main__":
    
    particle = 'p'
    n = 2
    
    if n==1: ############           SkX
        data=quickLoad("Densities/SkXDensityCa40"+particle+".dat")
        nucl_name = "Ca40"+particle+"_SkX"
            
    if n==2: ############           t0t3
        data=read("Densities/rho_ca40_t0t3.dat")
        nucl_name = "Ca40"+particle+"_t0t3"
        if particle == 'p': 
            data= data[0],data[1]
        else:
            data= data[0],data[2]
        
        
    for bound in [9., 10., 11., 12.]:
        print("using NO MOD \n")
        nucl0 = Problem(Z=20,N=20, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=data, \
                       output_folder=nucl_name, exact_hess=True)
            
        datas, info = nucl0.solve()
        datas = loadData(nucl0.datafile)
        status = datas['status']
        x = datas['x']
    
        solver0 = Solver(nucl0, x)
        x, check = solver0.solve()
        print (check)
        
        pot0 = solver0.getPotential()
        """
        print("using MOD \n")
        nucl1 = Problem_mod(Z=20,N=20, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityCa40p.dat"), \
                       output_folder="Ca40SkX", exact_hess=True)
            
        datas, info = nucl1.solve()
        datas = loadData(nucl1.datafile)
        status = datas['status']
        x = datas['x']
    
        solver1 = Solver(nucl1, x)
        x, check = solver1.solve()
        print (check)
        
        pot1 = solver1.getPotential()
        
        print("using MOD 2 \n")
        nucl2 = Problem_mod2(Z=20,N=20, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityCa40p.dat"), \
                       output_folder="Ca40SkX", exact_hess=True)
            
        datas, info = nucl2.solve()
        datas = loadData(nucl2.datafile)
        status = datas['status']
        x = datas['x']
    
        solver2 = Solver(nucl2, x)
        x, check = solver2.solve()
        print (check)
        
        pot2 = solver2.getPotential()
        
        
        #NO MOD:: ub=8 DO NOT converge, for ub >= 11 plots are terrible
        #MOD 2:: ub=8 max iter exceeded (still got a potential tho)
        """
        
        # Benchmark
        if n==1:
            out = read("Potentials\pot_ca40_skx.dat")
        elif n==2:
            out = read("Potentials\pot_ca40_t0t3.dat")
        if particle == 'p':
            r, vp = out[0], out[1]
        else:
            r, vp = out[0], out[2]

        # from C++ code
        if n==1:
            out = read("Potentials\pot_ca40"+particle+"_skx_other_iks.dat")
        elif n==2:
            out = read("Potentials\pot_ca40"+particle+"_t0t3_other_iks.dat")
        
        r_other, vp_other = out[0], out[1]
        
        
        #plotting results
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(solver0.grid, pot0 - pot0[60]+vp[60], label="CV acc=2")
        # ax.plot(solver1.grid, pot1 - pot1[3]+vp[3], label="CV acc=6")
        # ax.plot(solver2.grid, pot2 - pot2[3]+vp[3], label="CV acc=4")
        ax.plot(r, vp, '--', label="exact")
        ax.plot(r_other, vp_other - vp_other[46]+vp[60], label="other IKS")
        plt.grid(); plt.legend()
        
        ax.set_title(nucl_name + " " + str(bound) )#+ " mod 2")
        ax.set_xlabel("radius"),
        ax.set_ylabel("potential")
        ax.set_xlim([0, 10])
        ax.set_ylim([-60, 20])
    
    ##########################          RESULTS (ub = 10 which is the best)
    #NO MOD: 1870 iterations to converge at 330.726 [>>]
    #MOD 2: 580 iterations to converge at 330.723