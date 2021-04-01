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

--- must check if rho!=0 !! (see line 56)

ERROR: IKS does not converge for slightly different rho
"""
# GAIDUK APPROACH 

class Energy(object):
    """
    A class aimed to compute energies from a given potential or density.
    
    Parameters
    ----------
    problem : Problem
        class representing the IKS problem (see Problem)
    output : str
        folder directory inside Results/ in which the output is saved (default: Output)
    param_step : float
        step for the parametric intergal (default: 0.1)
    r_step : float 
        step for the radial integral (default: 0.1)
    R_min : float
        lower bound of the radial integral (default: 0.01)
    R_max : float
        upper bound of the radial integral (default: 10.)
    
    """
    
    def __init__(self, problem, output="Output", param_step=0.1, r_step=0.1, R_min=0.01, R_max=10.):
        
        self.dt = param_step
        self.dr = r_step
        self.T = np.arange(0.1, 1.1, self.dt) #parameter value list
        self.R = np.arange(R_min, R_max, self.dr)  #radius value list
        
        #------------ADD control over rho==0!! (np.any / np.all)
        self.rho_grid, self.rho = quickLoad(problem.file) if problem.data==[] else problem.data
        #print(self.rho_grid, self.rho)
        self.rho_fun = interpolate(self.rho_grid, self.rho)
        
        self.d_dx  = FinDiff(0, problem.h, 1, acc=4)

        self.problem = problem
        self.output = output
    
        
    
    """
    Return system energy w/ Q, K, L scaling
    
    Returns
    --------
    E : float []
        energy array from all scalings
        
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
    
    Parameters
    ----------
    v: float []
        matrix of potentials
    
    Returns
    ----------
    v: float []
        potential matrix evaluated in the integration radii 
        
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
    
    Returns
    ----------
    vQ: float []
        potential matrix for Q scaling
    vL: float []
        potential matrix for L scaling
    vZ: float []
        potential matrix for Z scaling
        
    """
    
    def IKS_Potential(self):
        v=[]
        self.status=[]
        for t in self.T: 
            print("\n\nComputing potential with parameter t: \t", t, '\n')
            
            #setting output directory for each minimization
            file = self.output + "/" + "minimization_t=" + str(t)
            
            #setting problem density with (r,t*rho)
            self.problem.setDensity(data=(self.rho_grid, t*self.rho), output_folder=file)
            
            #computing eigenfunctions
            res, info = self.problem.solve()
            x = res['x']
            self.status.append("Minimization with t = "+ str(t) + " : " + str(res['status']))
            
            #computing potentials
            solv = Solver(self.problem, x)
            v.append(solv.getPotential())
        
        
        t_col=np.reshape(self.T,newshape=(-1,1))
        #print("\n t_col \t", np.shape(t_col))
            
        return v, v/(3*t_col), v/(2*np.sqrt(t_col)), solv.grid
    
    
    """
    Integral calculation for Q,L,Z-scaling
    
    Parameters
    ----------
    rho: float []
        density  
    Drho: float []
        gradient density 
    vQ: float []
        potential matrix for Q scaling
    vL: float []
        potential matrix for L scaling
    vZ: float []
        potential matrix for Z scaling
    
    Returns
    ----------
    I_Q: float 
        energy value for Q scaling
    I_L: float 
        energy value for L scaling
    I_Z: float 
        energy value for Z scaling
        
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
    
    Parameters
    ----------
    E : float []
        energy array from all scalings
        
    """
    
    def _saveE(self, E):
        self.E_file = "Results/" + self.output + "/E.out"
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
    
    problem_IKS = Problem(Z=20,N=20, max_iter=2000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SkXDensityCa40p.dat"))
    #problem_IKS = Problem(Z=20,N=20, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat"))
    #problem_IKS = Problem(Z=82,N=126, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SOGDensityPb208p.dat"))
    
    energy = Energy(problem_IKS, "Ca40SkX_En")
    #energy = Energy(problem_IKS, "HO20coupled")
    #energy = Energy(problem_IKS, "Pb208SOG_En")
    
    E = energy.solver()
    print("Energies", E)