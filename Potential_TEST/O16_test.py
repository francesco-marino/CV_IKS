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

    particle = 'n'
    n = 1
    
    if n==1: ############           SkX
        data=quickLoad("Densities/SkXDensityO16"+particle+".dat")
        nucl_name = "O16"+particle+"_SkX"
            
    if n==2: ############           t0t3
        data=read("Densities/rho_o16_t0t3.dat")
        nucl_name = "O16"+particle+"_t0t3"
        if particle == 'p': 
            data= data[0],data[1]
        else:
            data= data[0],data[2]
        
        
    for bound in [8., 9., 10., 11., 12.]:
        print("using NO MOD \n")
        nucl0 = Problem(Z=8,N=8, n_type='p', max_iter=4000, ub=bound, debug='y', \
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
        nucl1 = Problem_mod(Z=8,N=8, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityO16p.dat"), \
                       output_folder="O16SkX", exact_hess=True)
            
        datas, info = nucl1.solve()
        datas = loadData(nucl1.datafile)
        status = datas['status']
        x = datas['x']
    
        solver1 = Solver(nucl1, x)
        x, check = solver1.solve()
        print (check)
        
        pot1 = solver1.getPotential()
        
        print("using MOD 2 \n")
        nucl2 = Problem_mod2(Z=8,N=8, n_type='p', max_iter=4000, ub=bound, debug='y', \
                       basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityO16p.dat"), \
                       output_folder="O16SkX", exact_hess=True)
            
        datas, info = nucl2.solve()
        datas = loadData(nucl2.datafile)
        status = datas['status']
        x = datas['x']
    
        solver2 = Solver(nucl2, x)
        x, check = solver2.solve()
        print (check)
        
        pot2 = solver2.getPotential()
        """
        # Benchmark
        if n==1:
            out = read("Potentials\pot_o16_skx.dat")
        elif n==2:
            out = read("Potentials\pot_o16_t0t3.dat")
        if particle == 'p':
            r, vp = out[0], out[1]
        else:
            r, vp = out[0], out[2]

        # from C++ code
        if n==1:
            out = read("Potentials\pot_o16"+particle+"_skx_other_iks.dat")
        elif n==2:
            out = read("Potentials\pot_o16"+particle+"_t0t3_other_iks.dat")
        
        r_other, vp_other = out[0], out[1]
            
        
        #plotting results
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(solver0.grid, pot0 - pot0[3]+vp[3], label="CV acc=2")
        # ax.plot(solver1.grid, pot1 - pot1[3]+vp[3], label="CV acc=6")
        # ax.plot(solver2.grid, pot2 - pot2[3]+vp[3], label="CV acc=4")
        ax.plot(r, vp, '--', label="exact")
        ax.plot(r_other, vp_other - vp_other[3]+vp[3], label="other IKS")
        plt.grid(); plt.legend()
        
        ax.set_title(nucl_name + " " +str(bound))
        ax.set_xlabel("radius"),
        ax.set_ylabel("potential")
        ax.set_xlim([0, 10])
        ax.set_ylim([vp[3]-10, 10])

    ########################## exact_hess=False         RESULTS (ub = 8 which is the best)
    #NO MOD: 690 iterations to converge at 122.537
    #MOD 2: 61O iterations to converge at 122.533 [>>]