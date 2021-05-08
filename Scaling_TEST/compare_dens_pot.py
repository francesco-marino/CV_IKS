# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:36:28 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Misc import read
from Orbitals import ShellModelBasis 
from Problem import Problem
from Solver import Solver
from Energy import Energy, quickLoad, interpolate
# from multiple_plots import PlotInColumns
from matplotlib import gridspec
import seaborn as sns


def PlotInColumns(x1,y1,x2,y2, title):
    
    fig = plt.figure(figsize=(13,16))
    
    #set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    sns.set_theme(style='white',font_scale=3)#, palette = 'Pastel2')

    #first plot
    ax1 = plt.subplot(gs[0])
    for i in range(len(x1)):
        ax1.plot(x1[i], y1[i], 
                 linewidth=2) 
    
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    #second plot
    ax2 = plt.subplot(gs[1], sharex = ax1)
    for i in range(len(x2)):
        ax2.plot(x2[i], y2[i], 
                 linewidth=2,
                 label = r"$\lambda =$" + str(param_list[i]) )
        
    plt.setp(ax1.get_xticklabels(), visible=False)
    # yticks = ax2.yaxis.get_major_ticks()
    # yticks[-1].label1.set_visible(False)
    
    ax1.set_xlim([0, 10.5])
    ax1.set_title(title)
    
    #labels
    ax2.set_xlabel("r")
    ax1.set_ylabel(r'$\rho_{\lambda}$(r)')
    ax2.set_ylabel(r"$v([\rho_{\lambda}]$,r)")
    
    #grid
    ax1.grid()
    ax2.grid()
    ax2.legend()
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 1.))#, borderaxespad=0.1)
    
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    
##############################################################################

prec = "_-8"
save = 'n'
n=3
if n==1:
    nucl_name = "O16t0t3"
    data = quickLoad("Densities/rho_o16_t0t3.dat")
    Z=8
    N=8
if n==1.5:
    nucl_name = "O16n_t0t3"
    data = read("Densities/rho_o16_t0t3.dat")
    data = data[0], data[2]
    Z=8
    N=8
elif n==2:
    nucl_name = "O16p_SkX"
    data = quickLoad("Densities/SkXDensityO16p.dat")
    Z=8
    N=8
elif n==3:
    nucl_name = "O16n_SkX"
    data = quickLoad("Densities/SkXDensityO16n.dat")
    Z=8
    N=8
elif n==4:
    nucl_name = "Ca40t0t3"
    data=quickLoad("Densities/rho_ca40_t0t3.dat")
    Z=20
    N=20
elif n==4.5:
    nucl_name = "Ca40n_t0t3"
    data=read("Densities/rho_ca40_t0t3.dat")
    data = data[0], data[2]
    Z=20
    N=20
elif n==5:
    nucl_name = "Ca40pSkX"
    data=quickLoad("Densities/SkXDensityCa40p.dat")
    Z=20
    N=20
elif n==6:
    nucl_name = "Ca40n_SkX"
    data=quickLoad("Densities/SkXDensityCa40n.dat")
    Z=20
    N=20
    
dens = interpolate(data[0], data[1])

def scaled_dens(r, rho, L):
    return L**3 * rho(L * r)

energy = Energy(data=data, C_code=True, \
                param_step=0.001, t_min=0.9, t_max=1.0, \
                input_dir="Scaled_Potentials/" + nucl_name + prec, scaling='l')

energy.getPotential_En()

elim = np.arange(-10,0,1)
status = []
cutoff = 1e-9

#to plot
x1C=[];y1C=[]
x2C=[];y2C=[]
x1P=[];y1P=[]
x2P=[];y2P=[]

param_list = [0.4, 0.7, 1., 1.3]
for t in param_list:
    # C++
    name_pot = "/Potentials/pot_L=" + str('%.2f'%t) + "0000_C++.dat"
    rpC, pC = quickLoad(energy.input + name_pot, beg=3, end=12)
    pC = energy.shiftPotentials(rpC, pC)
    
    name_dens = "/Densities/den_L=" + str('%.2f'%t) + "0000_C++.dat"
    rdC, dC = quickLoad(energy.input + name_dens, beg=3, end=12)
    
    
    # Python
    rho = lambda r : scaled_dens(r, dens, t)
    
    for rr in np.arange(0.,50.,0.1):
        if( rho(rr) < cutoff ):
            bound = rr - 0.1
            break
    # print(bound)
    print("lambda: ", t)
    nucl = Problem(Z, N, n_type='p', max_iter=4000, ub=bound, debug='y', \
                      basis=ShellModelBasis(), rho=rho,\
                      exact_hess=True, output_folder="trials")
    results, info = nucl.solve()
    
    status.append(results['status'])
    
    solver = Solver(nucl)
    x, check = solver.solve()
    
    gridL = solver.grid
    potL = solver.getPotential()
    
    gridL = np.delete(gridL, elim)
    potL = np.delete(potL, elim)

    rdP=np.arange(0.,solver.grid[-1],0.1)
    dP=rho(rdP)
    
    x1C.append(rdC)
    y1C.append(dC)
    x2C.append(rpC)
    y2C.append(pC - pC[-8])#- pC[40]
    
    x1P.append(rdP)
    y1P.append(dP)
    x2P.append(gridL)
    y2P.append(potL - potL[-8]) #- potL[75]
    
"""
#C++
PlotInColumns(rdC, dC, rpC, pC)
#Python
PlotInColumns(rdP, dP, gridL, potL)
"""


    
#C++
PlotInColumns(x1C, y1C, x2C, y2C, nucl_name+prec+" C++")
#Python
PlotInColumns(x1P, y1P, x2P, y2P, nucl_name+prec+" Python")

if save=='y':
    file_out = "Scaling_TEST/"+ nucl_name + prec +"/Summary.dat"
    
    status = np.reshape(status, newshape=(-1,1))
    save = np.column_stack((energy.T,status))
    np.savetxt(file_out, save, delimiter='  ', header='string', comments='', fmt='%s')
