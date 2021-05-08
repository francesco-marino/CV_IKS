# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:36:28 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Orbitals import ShellModelBasis 
from Problem import Problem
from Solver import Solver
from Energy import Energy, quickLoad, interpolate

save='n'
prec = "_-8"
n=4
if n==1:
    nucl_name = "O16t0t3"
    data = quickLoad("Densities/rho_o16_t0t3.dat")
    Z=8
    N=8
elif n==2:
    nucl_name = "O16SkX"
    data = quickLoad("Densities/SkXDensityO16p.dat")
    Z=8
    N=8
elif n==3:
    nucl_name = "Ca40t0t3"
    data=quickLoad("Densities/rho_ca40_t0t3.dat")
    Z=20
    N=20
elif n==4:
    nucl_name = "Ca40pSkX"
    data=quickLoad("Densities/SkXDensityCa40p.dat")
    Z=20
    N=20
    
dens = interpolate(data[0], data[1])

def scaled_dens(r, rho, L):
    return L**3 * rho(L * r)

energy = Energy(data=data, C_code=True, \
                param_step=0.001, t_min=0.9, t_max=1.0, \
                input_dir="Scaled_Potentials/" + nucl_name + prec, scaling='l')

energy.solver()

elim = np.arange(-10,0,1)
status = []
cutoff = 1e-9

for t in energy.T:
    # C++
    name = "/Potentials/pot_L=" + str('%.2f'%t) + "0000_C++.dat"
    r, p = quickLoad(energy.input + name, beg=3, end=12)
    p = energy.shiftPotentials(r, p)
    
    
    # Python
    rho = lambda r : scaled_dens(r, dens, t)
    
    for rr in np.arange(0.,50.,0.1):
        if( rho(rr) < cutoff ):
            bound = rr - 0.1
            break
    # print(bound)
    print("lambda: ", t)
    nucl = Problem(Z, N, n_type='p', max_iter=2000, ub=bound, debug='y', \
                      basis=ShellModelBasis(), rho=rho,\
                      exact_hess=True, output_folder=nucl_name+prec)
    results, info = nucl.solve()
    
    status.append(results['status'])
    
    solver = Solver(nucl)
    x, check = solver.solve()
    
    gridL = solver.grid
    potL = solver.getPotential()
    
    gridL = np.delete(gridL, elim)
    potL = np.delete(potL, elim)

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(
        r, p,
        label = "C ++",
        # ls = '--',
        lw = 2
        )
    ax.plot(
        gridL, potL - potL[-15],
        label = "Python",
        lw = 2
        )
    plt.grid(); plt.legend()
    ax.set_title("Potentials with lambda " + str(t) + nucl_name + prec)
    ax.set_xlabel("Radius r")
    # ax.set_xlim([0, 11])
    # ax.set_ylim([-50, -100])
    ax.set_ylabel("Potential")

if save=='y':
    file_out = "Scaling_TEST/"+ nucl_name + prec +"/Summary.dat"
    
    status = np.reshape(status, newshape=(-1,1))
    save = np.column_stack((energy.T,status))
    np.savetxt(file_out, save, delimiter='  ', header='string', comments='', fmt='%s')
