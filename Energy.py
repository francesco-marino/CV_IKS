# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:34:21 2021

@author: alberto
"""

import numpy as np
import os
from findiff import FinDiff

from Problem import Problem, quickLoad
from Solver import Solver
from Orbitals import ShellModelBasis

"""
--- think about: domain of the energy since potentials nearby r=0 are not trustworthy

--- add integral precision from input and integral error evaluation
    -> change integral method

--- problem with basis and self.basis

--- must check if rho!=0 !! (see line 44)

ERROR: IKS does not converge for slightly different rho
"""
# GAIDUK APPROACH 

class Energy(object):
    
    def __init__(self, Z, N=0, rho=None, lb=0.1, ub=10., h=0.1, n_type="p",\
                 data=[], basis=ShellModelBasis(), max_iter=2000, rel_tol=1e-3,\
                 constr_viol=1e-3, debug='n', file="Densities/SkXDensityCa40p.dat",\
                 output="Output", param_step=0.1, r_step=0.1):
        
        self.dt = param_step
        self.dr = r_step
        self.h = h
        self.T = np.arange(0.1, 1.1, self.dt) #parameter value list
        self.R = np.arange(0.1, 10., self.dr)  #radius value list
        
        #density from data: both array and function
        #------------ADD control over rho==0!! (np.any / np.all)
        self.rho_grid, self.rho = quickLoad(file) if data==[] else data
        #print(self.rho_grid, self.rho)
        self.rho_fun = interpolate(self.rho_grid, self.rho)
        
        self.d_dx  = FinDiff(0, self.h, 1, acc=4)

        #saving parameters for IPOPT minimization
        self.output = output
        self.Z = Z
        self.N = N 
        self.max_iter = max_iter
        self.debug = debug
        #self.basis = basis
        
        self.rel_tol = rel_tol 
        self.constr_viol = constr_viol
        
    
    """
    Return system energy w/ Q, K, L scaling
    """
    
    def solver(self):
        
        # computing potentials with IKS
        self.vQ, self.vL, self.vZ, self.v_grid = self.IKS_Potential()
        
        #density for given r of the integral
        rho_int = self.rho_fun(self.R)
        Drho_int= self.d_dx(rho_int)
        #potential for given r of the integral
        v_int_Q = self._evalPotential(self.vQ)
        v_int_L = self._evalPotential(self.vL)
        v_int_Z =  self._evalPotential(self.vZ)
        #print("v Q",v_int_Q, "\n\n L", v_int_L, "\n\n Drho", Drho_int, "\n\n Z", v_int_Z)
        
        E = self.calcIntegral(rho_int, Drho_int, v_int_Q, v_int_L, v_int_Z)
        
        self._saveE(E)
    
        return E
    
    
    """
    Evaluate potential within the integration grid
    """
    
    def _evalPotential(self, v):
        #get continuous functions of potential
        v_fun=[]
        for i in range(len(v)):
            v_fun.append(interpolate(self.v_grid, np.array(v)[i,:]))
            
        #potential evaluation for the integral
        v_int=[]
        for j in range(len(v_fun)):
            v_int.append(v_fun[j](self.R))
        
        return v_int
    
    
    """
    Parametric potentials from IKS
    """
    
    def IKS_Potential(self):
        v=[]
        self.status=[]
        for t in self.T: 
            print("\n\nComputing potential with parameter t: \t", t, '\n')
            #defining the problem with (r,t*rho). it is an array and not a function
            problem = Problem(self.Z, self.N, data=(self.rho_grid, t*self.rho),\
                              max_iter=self.max_iter, debug=self.debug,\
                              basis=ShellModelBasis(), rel_tol=self.rel_tol,\
                              constr_viol=self.constr_viol, output_folder=self.output)
            #print(self.basis)
            #basis=ShellModelBasis()
            #print(self.basis == basis)
            #problem=Problem(20,20,data=(self.rho_grid, t*self.rho),max_iter=4000, debug='y',\
            #                basis=basis, rel_tol=1e-4, constr_viol=1e-4, output_folder="Ca40SkX_En"  )
            #computing and getting results
            res, info = problem.solve()
            x = res['x']
            self.status.append("Minimization with t = "+ str(t) + " : " + str(res['status']))
            #computing and saving potentials
            solv = Solver(problem, x)
            v.append(solv.getPotential())
        
        self.output = problem.output_folder
        t_col=np.reshape(self.T,newshape=(-1,1))
        #print("\n t_col \t", np.shape(t_col))
            
        return v, v/(3*t_col), v/(2*np.sqrt(t_col)), solv.grid
    
    
    """
    Integral calculation for Q,L,Z-scaling
    """
    
    def calcIntegral(self, rho, Drho, vQ, vL, vZ):
        print("SHAPES: \n rho \t", np.shape(rho), "\n Drho \t", np.shape(Drho),\
              "\n vQ \t", np.shape(vQ), "\n vL \t", np.shape(vL), "\n vZ \t", np.shape(vZ))
        
        #4Pi r^2 rho v
        f_Q = np.sum(4*np.pi*self.R**2*rho*vQ)
        I_Q = f_Q*self.dt*self.dr
        #4Pi r^2 ( 3 rho + r grad(rho) ) v
        f_L = np.sum(4*np.pi*self.R**2*(3*rho+self.R*Drho)*vL)
        I_L = f_L*self.dt*self.dr
        #4Pi r^2 ( 2 rho + r/3 grad(rho) ) v
        f_Z = np.sum(4*np.pi*self.R**2*(2*rho+self.R/3.*Drho)*vZ)
        I_Z = f_Z*self.dt*self.dr
        
        return I_Q, I_L, I_Z
    
    
    """
    Printing energy on file
    """
    
    def _saveE(self, E):
        self.E_file = self.output + "/E.out"
        with open(self.E_file, 'w') as fo:
            status = np.reshape(self.status, newshape=(-1,1))
            fo.write(str(status))
            fo.write("\n \n ENERGIES: \n Q-path:\t" + str(E[0]))
            fo.write("\n Lambda-path:\t" + str(E[1]))
            fo.write("\n Z-path:\t" + str(E[2]))
    
    
    """
    Returnig the potential
    """
    
    def _getPot(self):
        return self.v_grid, self.vQ, self.vL, self.vZ
    
    
"""
Interpolation of a function f in its (discrete) domain r
"""
    
def interpolate(r, f):
    from scipy import interpolate
    
    r=np.array(r); f=np.array(f);
    assert( len(r)==len(f) )
     
    tck  = interpolate.splrep(r, f)
    ff = lambda x: interpolate.splev(x, tck )
        
    return ff    

#########################################################
########################TEST#############################
#########################################################

if __name__ == "__main__":
    
    #file = "Densities/SOGDensityPb208p.dat"
    #file = "Densities/SkXDensityCa40p.dat"
    file = "Densities/rho_HO_20_particles_coupled_basis.dat"
    energy = Energy(Z=20,N=0, max_iter=10, rel_tol=1e-4, constr_viol=1e-4, param_step=0.1, r_step=0.1, file=file, output="HO20coupled")
    #energy = Energy(Z=20,N=20, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, param_step=0.1, r_step=0.1, file=file, output="Ca40SkX_En")
    #energy = Energy(Z=82,N=126, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, param_step=0.1, r_step=0.1, file=file)#, output="Pb208SOG_En")
    E = energy.solver()
    print("Energies", E)