# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:34:21 2021

@author: alberto
"""

import numpy as np
import os
from findiff import FinDiff
from scipy import integrate
import matplotlib.pyplot as plt

from Problem import Problem, quickLoad
from Solver import Solver

"""
TO DO/CHECK LIST: 
    
--- think about: domain of the energy since IKS potentials nearby r=0 are not trustworthy

--- give accessibility: choose which scaling one wants to do (could be awful to read)\
                        whether to save Energy value on file

--- check where rho==0 (see line 74) (ex HO_20_particles_coupled_basis), it gives problem somewhere

--- split computation of Q/Lambda/Z path? better readable 

--- merge energy evaluation for different R_max into the class (as in Test_case0/1/2) (??)

ERROR: IKS does not converge for slightly different rho
"""
# GAIDUK APPROACH 

class Energy(object):
    """
    A class aimed at computing energies from a given potential and density or from an IKS problem.
    
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
    
    def __init__(self, problem=None, rho=None, v=None, grad_rho=None, output="Output",\
                 param_step=0.1, r_step=0.1, R_min=0.01, R_max=10., integrator=integrate.simpson,\
                 scaling="all", initial_rho=None):
        
        self.dt = param_step
        self.dr = r_step
        self.T = np.arange(self.dt, 1. + self.dt, self.dt) #parameter value list
        self.R = np.arange(R_min, R_max + self.dr, self.dr)  #radius value list
        
        self.d_dx  = FinDiff(0, self.dr, 1, acc=4)
        
        assert not(problem==None and (rho==None or v==None)) #is ValueError better?
        
        #if both methods are present, IKS problem comes first
        if (problem!=None):
            #IKS problem
            
            #------------ADD control over rho==0!! (np.any / np.all)
            self.rho_grid, self.rho = quickLoad(problem.file) \
                if problem.data==[] else problem.data
            #print(self.rho_grid, self.rho)
            
            self.rho_fun = interpolate(self.rho_grid, self.rho)
            
            #saving problem and method
            self.problem = problem
            self.method = "IKS"
            
        else: 
            #Input-given density and potential
            self.rho_fun = rho
            self.rho = self.rho_fun(self.R)
            self.v_fun = v
            
            if (grad_rho!=None):
                self.grad_rho_fun = grad_rho
                self.grad_rho = grad_rho(self.R)
                
            else: 
                #if none is given, computing the gradient
                self._setGradient()
            
            if len(output)>0 and not os.path.exists("Results/" + output):
                os.makedirs("Results/" + output)
                
            
            self.method = "Rho and v from input"
            
        #saving output directory
        self.output = output
        #saving integration method
        self.integrator = integrator
        
        #defining normalization
        self.N = 4*np.pi*self.integrator(self.R**2 * self.rho, self.R)
      
        
    """
    Return system energy w/ Q, K, L scaling
    
    Returns
    --------
    E : float []
        energy array from all scalings
        
    """
    
    def solver(self):
        
        if (self.method == "IKS"):
            # computing potentials with IKS
            vQ, vL, vZ, v_grid = self.IKS_Potential()
            
            #evaluate potentials in self.R (needed in the integral)
            self.vQ = self._evalPotential(vQ, v_grid, 'q')
            self.vL = self._evalPotential(vL, v_grid, 'l')
            self.vZ = self._evalPotential(vZ, v_grid, 'z')

            #getting density and its gradient in self.R
            self.rho = self.rho_fun(self.R)
            self.grad_rho = self.d_dx(self.rho)

        else:
            #computing potentials given from input
            self.vQ, self.vL, self.vZ = self.Input_Potential()

            
        #print("v Q",self.vQ, "\n\n L", self.vL, "\n\n Z", self.vZ, "\n\n Drho", self.grad_rho)
        E = self.calcIntegral()
        
        self._saveE(E)
    
        return E
    
    
    """
    Parametric potentials from IKS
    
    Parameters
    ----------
    
    Returns
    ----------
    vQ: float []
        potential matrix for Q scaling
    vL: float []
        potential matrix for L scaling
    vZ: float []
        potential matrix for Z scaling
    solv.grid: float []
        grid in which potentials are evaluated
        
    """
    
    def IKS_Potential(self):
        
        for scaling in ["q", "lambda", "z"]:
            v=[]
            self.status=[]
            
            for t in self.T: 
                
                if scaling == "q":
                    x = self.rho_grid
                    Rho = self.rho * t
                elif scaling == "lambda":
                    x = self.rho_grid * t
                    Rho = self.rho * t**3
                elif scaling == "z":
                    x = self.rho_grid * t**(1./3.)
                    Rho = self.rho * t**2
                
                print("\n\nComputing potential with ", scaling, \
                      " scaling and parameter t: \t", t, '\n')
                
                #setting output directory for each minimization
                file = self.output + "/" + scaling + "_minimization_t=" + str('%.1f'%t)
                
                #setting problem density with (r,rho_t(r))
                self.problem.setDensity(data=(x, Rho), output_folder=file)
                
                #computing eigenfunctions
                res, info = self.problem.solve()
                x = res['x']
                
                st = "Minimization with t = "+ str('%.1f'%t) + " : " + str(res['status'])
                self.status.append(st)
                
                #computing potentials
                solv = Solver(self.problem, x)
                v.append(solv.getPotential())
            
            #saving potentials
            if scaling == "q":
                self.status_Q = self.status
                vQ = v
            elif scaling == "lambda":
                self.status_L = self.status
                vL = v
            elif scaling == "z":
                self.status_Z = self.status
                vZ = v
                    
        return vQ, vL, vZ, solv.grid 
        # they are all evaluated in solv.grid
        
        
    """
    Evaluate potential within the integration grid
    
    Parameters
    ----------
    v: float []
        matrix of potentials
    
    Returns
    ----------
    v: float []
        matrix of potentials evaluated in the integration radii 
        
    """
    
    def _evalPotential(self, v, v_grid, scaling):
        
        v_int=[]
            
        for j in range(len(self.T)):
            t = self.T[j]
            
            if scaling == "q":
                x = self.R
            elif scaling == "l":
                x = self.R * t
            elif scaling == "z":
                x = self.R * t**(1./3.)
            
            #get continuous functions of potential (one row of the matrix)
            v_fun = interpolate(v_grid, np.array(v)[j,:])
            #potential evaluation in the integral grid
            v_int.append(v_fun(x))
          
        return v_int
    
    
    """
    to do
    computing parametric potential given from input
    """
    
    def Input_Potential(self): 
        
        vQ=[]; vL=[]; vZ=[]
        
        for t in self.T:
            # Q
            row = self.v_fun(self.R, self.rho_fun, self.N, t)
            vQ.append(row)
            # L
            x = self.R * t 
            row = self.v_fun(x, self.rho_fun, self.N, t**3)
            row = row
            vL.append(row)
            # Z
            x = self.R * t**(1./3.)
            row = self.v_fun(x, self.rho_fun, self.N, t**2)
            vZ.append(row)
        
        return np.array(vQ), np.array(vL), np.array(vZ)
    
    
    """
    to do 
    
    Integral calculation for Q,L,Z-scaling
    
    Parameters
    ----------
    
    Returns
    ----------
    I_Q: float 
        energy value for Q scaling
    I_L: float 
        energy value for L scaling
    I_Z: float 
        energy value for Z scaling
        
    """
    
    def calcIntegral(self):
        # print("INTEGRAL, SHAPES: \n rho \t", np.shape(self.rho), "\n Drho \t", \
        # np.shape(self.grad_rho), "\n vQ \t", np.shape(self.vQ), \
        # "\n vL \t", np.shape(self.vL), "\n vZ \t", np.shape(self.vZ))

        #Q integral
        I = self.integrator(self.vQ, self.T, axis=0) #integral over t
        f_Q = I * 4 *np.pi * self.R**2 * self.rho / self.N
        I_Q = self.integrator(f_Q, self.R) #integral over r
        
        #LAMBDA integral
        I_r=[]
        for j in range(len(self.T)):
            # L
            l = self.T[j]
            x = self.R * l
            f_L = self.vL[j,:] / l * 4 * np.pi * x**2 * \
                (3 * self.rho_fun(x) / self.N + x * self.grad_rho_fun(x) / self.N)
            I = self.integrator(f_L, x) #integrating over r
            I_r.append(I)
            
        I_L = self.integrator(I_r, self.T) #integrating over t
        
        #Z integral
        I_r=[]
        for j in range(len(self.T)):
            # Z
            z = self.T[j]
            x = self.R * z**(1./3.) 
            f_Z = self.vZ[j,:] * 4 * np.pi * x**2 * \
                (2 * self.rho_fun(x) / self.N + x/3. * self.grad_rho_fun(x) / self.N)
            I = self.integrator(f_Z, x) #integrating over r
            I_r.append(I)
        
        I_Z = self.integrator(I_r, self.T) #integrating over t
        
        
        return I_Q, I_L, I_Z
    
    
    """
    to do 
    setting the density gradient function
    """
    def _setGradient(self):
        
        self.grad_rho = self.d_dx(self.rho)
        self.grad_rho_fun = interpolate(self.R, self.grad_rho)
            
        
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
            if(self.method == "IKS"):
                status = np.reshape(self.status, newshape=(-1,1))
                fo.write(str(status))
            fo.write("\n \n ENERGIES: \n Q-path:\t" + str(E[0]) + "\n\n")
            fo.write("\n Lambda-path:\t" + str(E[1]) + "\n\n")
            fo.write("\n Z-path:\t" + str(E[2]) + "\n\n")
    
    
    """
    Returnig the potential
    """
    
    def _getPot(self):
        return self.R, self.vQ, self.vL, self.vZ
    
    
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
    
    # problem_IKS = Problem(Z=20,N=20, max_iter=10, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SkXDensityCa40p.dat"))
    #problem_IKS = Problem(Z=20,N=20, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat"))
    #problem_IKS = Problem(Z=82,N=126, max_iter=4000, rel_tol=1e-4, constr_viol=1e-4, data=quickLoad("Densities/SOGDensityPb208p.dat"))
    
    #energy = Energy(problem_IKS, "Ca40SkX_En")
    #energy = Energy(problem_IKS, "HO20coupled")
    #energy = Energy(problem_IKS, "Pb208SOG_En")
    
    print("\n \n TEST 1: ------------ \n \n")
    #TEST 1
    rho = lambda r : np.ones_like(r)
    grad_rho = lambda r : np.zeros_like(r)
    # potential = lambda rho : rho
    def potential(r, rho, N=1, t=1):
        v = t * rho(r) / N

        return v
    
    # energy = Energy(rho=rho, v=potential, grad_rho=grad_rho, param_step=0.001, r_step=0.001, R_min=0., R_max=10.)
    energy = Energy(rho=rho, v=potential, param_step=0.001, r_step=0.001, R_min=0., R_max=10.)
    
    E = energy.solver()
    print("Energies: \n",\
          "\n\nEnergies with trapezoids: ", E[0],\
          "\n\nEnergies from simpson: ", E[1])
        
    
    print("\n \n TEST 2: ------------ \n \n")
    #TEST 2   
    rho = lambda r : r
    grad_rho = lambda r : np.ones_like(r)
    # potential = lambda rho : rho**2
    # potential = lambda rho,r : rho(r)**2
    
    def potential(r, rho, N=1, t=1):
        v = (t * rho(r) / N) **2

        return v
        
    # energy = Energy(rho=rho, v=potential, grad_rho=grad_rho, param_step=0.001, r_step=0.01, R_min=0., R_max=5.)
    energy = Energy(rho=rho, v=potential, param_step=0.001, r_step=0.001, R_min=0., R_max=5.)
    
    
    E = energy.solver()
    print("Energies: \n",\
          "\n\nEnergies with trapezoids: ", E[0],\
          "\n\nEnergies from simpson: ", E[1])
