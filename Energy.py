# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:34:21 2021

@author: alberto
"""

import numpy as np
import os
from findiff import FinDiff
from scipy import integrate
import scipy.interpolate as scint
import matplotlib.pyplot as plt

# from Problem import Problem, quickLoad
from Solver import Solver

"""
TO DO/CHECK LIST: 
    
--- NB: IKS potentials nearby r=0 are not trustworthy

--- give accessibility: choose which scaling one wants to do (could be awful to read)\
                        whether to save Energy value on file

--- check where rho==0 (see line 74) (ex HO_20_particles_coupled_basis), it gives problem somewhere

--- split computation of Q/Lambda/Z path? better readable 

ERROR: IKS is still not stable enough for scaled densities and potentials
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
    
    def __init__(self, problem=None, rho=None, v=None, grad_rho=None, data=[], output="Output",\
                 param_step=0.1, t_min=0., t_max=1., r_step=0.1, R_min=1e-4, R_max=10., \
                 integrator=integrate.simpson, scaling="all", C_code=False, input_dir="Scaled_Potentials"):
        
        #NOTES:: 
            #self.R must be chosen in order to consider all potentials after they have gone to 0. (IKS)
            #also self.T must be set from the IKS potentials (v. checkCPotentials())
        
        self.dr = r_step
        num = int((R_max - R_min) / self.dr)
        self.R = np.linspace(R_min, R_max, num)  #radius value list
        
        self.d_dx  = FinDiff(0, r_step, 1, acc=4)
        
        #is ValueError better in assert errors?
        
        # assert not(problem==None and ((rho==None and len(data)==0) or v==None) \
                   # and (len(data)==0 or C_code==False)) 

        
        #if both methods are present, IKS problem comes first
        
        #IKS problem
        if (problem is not None):
            
            #------------ADD control over rho==0!! (np.any / np.all)
            self.rho_grid, self.rho = quickLoad(problem.file) \
                if problem.data==[] else problem.data
            #print(self.rho_grid, self.rho)
            
            self.rho_fun = interpolate(self.rho_grid, self.rho)
            
            #saving problem and method
            self.problem = problem
            self.method = "IKS python"
        
        # using provided density
        elif(rho is not None or len(data)>0): 
            assert( v is not None or C_code is True ) 
            
            # Input density function
            if(rho!=None):
                self.rho_fun = rho
                self.rho = self.rho_fun(self.R)
                
                # provided density gradient
                if (grad_rho!=None):
                    self.grad_rho_fun = grad_rho
                    self.grad_rho = grad_rho(self.R)
                
                # numerical gradient (if none is given)
                else: 
                    self.grad_rho = self.d_dx(self.rho)
                    self.grad_rho_fun = interpolate(self.R, self.grad_rho)
            
            # Input density array
            else:
                self.rho_fun, self.grad_rho_fun = interpolate(data[0], data[1], der=True)
                self.rho = self.rho_fun(self.R)
                self.grad_rho = self.grad_rho_fun(self.R)
                
            self.method = "Rho and v from input"
            
            # Input potential function
            if(v!=None):
                num = int((t_max - t_min) / param_step)
                self.T = np.linspace(t_min, t_max, num) #parameter value list
                
                self.v_fun = v
                
            # Using datas from C++ code
            else:
                self.t_min = t_min
                self.t_max = t_max
                self.dt = param_step
                self.input = input_dir
                
                self.method = "IKS C++"
            
            
            if len(output)>0 and not os.path.exists("Results/" + output):
                    os.makedirs("Results/" + output)
        
        else:
            assert(False), "Invalid input parameters"
            
            
        #saving output directory
        self.output = output
        #saving integration method
        self.integrator = integrator
        #saving scaling preference
        self.scaling = scaling
        
        #defining normalization
        self.N = 4*np.pi*self.integrator(self.R**2 * self.rho, self.R)
      
        
    def cutoff(rho):
        r=0.
        while(True):
            if( rho(r) < 1e-8 ):
                return r - 0.1
            r += 0.1
        
    """
    Return system energy w/ Q, K, L scaling
    
    Returns
    --------
    E : float []
        energy array from all scalings
        
    """
    
    def solver(self):
        
        # computing potentials with IKS
        if (self.method == "IKS"):
            
            vQ, vL, vZ, v_grid = self.IKS_Potential()
            
            #evaluate potentials in self.R (needed in the integral)
            self.vQ = self._evalPotential(vQ, v_grid, 'q')
            self.vL = self._evalPotential(vL, v_grid, 'l')
            self.vZ = self._evalPotential(vZ, v_grid, 'z')

            #getting density and its gradient in self.R
            self.rho = self.rho_fun(self.R)
            self.grad_rho = self.d_dx(self.rho)
        
        #computing potentials given from input
        elif(self.method == "Rho and v from input"):
            
            self.vQ, self.vL, self.vZ = self.Input_Potential()
        
        else:
            
            self.vL = self.C_Potential()
            
            
        #print("v Q",self.vQ, "\n\n L", self.vL, "\n\n Z",\
            #self.vZ, "\n\n Drho", self.grad_rho)
        
        # Compute the full (parametric + spatial) integral
        E = self.calcIntegral()
        # Save results to file
        self._saveE(E)
    
        return E
        
        
    """
    Parametric potentials from IKS.
    Return IKS potentials for different values of the "t" parameter.
    
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
                #NB -------- NOT TRUE! can't use self.rho since density mut be evaluated in x! 
                #NB -------- Nevertheless, when passing the grid to problem this must be self.R and NOT x.
                # ... since the scaled density is still a function of r only
                
                #it should be like the following: (to check)
                """
                if scaling == "q":
                    x = self.rho_grid
                    Rho = self.rho * t
                elif scaling == "lambda":
                    x = self.rho_grid * t
                    Rho = self.rho_fun(x) * t**3
                elif scaling == "z":
                    x = self.rho_grid * t**(1./3.)
                    Rho = self.rho_fun(x) * t**2
                """
                print("\n\nComputing potential with ", scaling, \
                      " scaling and parameter t: \t", t, '\n')
                
                #setting output directory for each minimization
                file = self.output + "/" + scaling + "_minimization_t=" + str('%.1f'%t)
                
                #setting problem density with (r,rho_t(r))
                #NB ---------- pay attention to the grid!!
                self.problem.setDensity(data=(self.rho_grid, Rho), output_maxolder=file)
                
                #computing eigenfunctions
                res, info = self.problem.solve()
                x = res['x']
                
                st = "Minimization with t = "+ str('%.1f'%t) + " : " + str(res['status'])
                self.status.append(st)
                
                #computing potentials
                # solv = Solver(self.problem, x) #no need of x
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
                
       # they are all evaluated in solv.grid              
        return vQ, vL, vZ, solv.grid 
       
        
        
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
            row = self.v_fun(x, self.rho_fun, self.N, t**3, self.R)
            row = row
            vL.append(row)
            # Z
            x = self.R * t**(1./3.)
            row = self.v_fun(x, self.rho_fun, self.N, t**2, self.R)
            vZ.append(row)
        
        return np.array(vQ), np.array(vL), np.array(vZ)
    
    
    """
    Load and calculate scaled potentials from C++ files
    """
    
    def C_Potential(self):
        # radial grid is maximum for the smallest lambda, \
            # all potentials will be evaluated there
        
        # create radial grid
        name = "/Potentials/pot_L=" + str('%.2f'%self.t_min) + "0000_C++.dat"
        r, p = quickLoad(self.input + name)
        
        num = int(r[-1] / self.dr)
        self.R = np.linspace(0., r[-1], num)
        # print(r[-1])
        
        # create parametrical grid
        self.checkCPotentials()
    
        self.vL=[]
        for t in self.T:
        # for i, t in enumerate(self.T): ##use this for plots below
            name = "/Potentials/pot_L=" + str('%.2f'%t) + "0000_C++.dat"
            r, p = quickLoad(self.input + name, beg=3, end=3)
            p = self.shiftPotentials(r, p)
            r, p = self.extendPotentials(r, p)
            
            # plot(r, p, t)
            
            # v_fun = interpolate(r, p) ## with spline
            v_fun = scint.interp1d(r, p, fill_value="extrapolate") ## with interp1d
            self.vL.append(v_fun(self.R))
            
            # plot(self.R, self.vL[i], t)
        
        return np.array(self.vL)
    
    
    """
    Create parametrical grid while checking convergence of C++ potentials
    """
    
    def checkCPotentials(self):
        self.T = np.arange(self.t_min, self.t_max, self.dt) # + self.dt
        if (self.t_max-self.T[-1])>1e-6 : self.T = np.append(self.T, self.t_max) # (*)
        
        b, l = quickLoad(self.input + "/Status.dat")
        elim = np.array(np.where(b!=1))
        self.T = np.delete(self.T, elim)
        
        # print(np.arange(0.43,1.,0.01))
        # PROBLEM: why the heck does it end in 1.?!
        # that is why (*) is required
    
    
    """
    Shifting potentials so that v(r)=0 for r->inf
    """
    
    def shiftPotentials(self, rad, pot):
        # DeltaPot = np.amax(pot) - np.amin(pot)
        pot = pot - pot[-5]
        
        return pot
        
    
    """
    Extend potentials with 0 till the radial grid end
    """
    
    def extendPotentials(self, rad, pot):
        R = rad[-1]
        while (R < self.R[-1]):
            # print(t)
            R += 0.1
            rad = np.append(rad, R)
            pot = np.append(pot, 0.)
            
        return rad, pot
    
    
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
        I_Q=[]; I_L=[]; I_Z=[];
        
        
        if self.scaling == "all" or self.scaling == "q" :
            #Q integral
            
            #integral over t
            I = self.integrator(self.vQ, self.T, axis=0) 
            f_Q = I * 4 *np.pi * self.R**2 * self.rho / self.N
            #integral over r
            I_Q = self.integrator(f_Q, self.R) 
        
        
        if self.scaling == "all" or self.scaling == "l" :
            #LAMBDA integral
            
            I_r=[]
            self.integrand=[]
            for j, l in enumerate(self.T):
                # L
                x = self.R * l
                f_L = self.vL[j,:] / l * 4 * np.pi * x**2 * \
                    (3 * self.rho_fun(x) / self.N + x * self.grad_rho_fun(x) / self.N)
                #integrating over r
                I = self.integrator(f_L, x)
                I_r.append(I)
                
                self.integrand.append(f_L)
                
            #integrating over t
            I_L = self.integrator(I_r, self.T) 
        
        
        if self.scaling == "all" or self.scaling == "z" :
            #Z integral
            
            I_r=[]
            for j, z in enumerate(self.T):
                # Z
                x = self.R * z**(1./3.) 
                f_Z = self.vZ[j,:] * 4 * np.pi * x**2 * \
                    (2 * self.rho_fun(x) / self.N + x/3. * self.grad_rho_fun(x) / self.N)
                #integrating over r
                I = self.integrator(f_Z, x)
                I_r.append(I)
            
            #integrating over t
            I_Z = self.integrator(I_r, self.T)
        
# @F: why doing I_r=np.zeros_like(...) and I_r[j]=... instead of I_r.append(...) ?
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
    
def interpolate(r, f, der=False):
    from scipy import interpolate
    
    r=np.array(r); f=np.array(f);
    assert( len(r)==len(f) )
     
    tck  = interpolate.splrep(r, f)
    ff = lambda x: interpolate.splev(x, tck )
    
    if(der == False):
        return ff    
    else: 
        d1 = lambda x: interpolate.splev(x, tck, der=1)
        
        return ff, d1


"""
Read datas from file
"""

def quickLoad(file, beg=0, end=0):
        count = 0
        file = open(file)
        ff = file.readlines()
        file.close()
        r = []; dp = []
        for ll in ff:
            if str(ll).startswith("#"):
                  pass
            elif(count < beg):
                count += 1
                pass
            elif(ll==ff[-end] and end!=0):
                break
            else:
                ll = [ float(x) for x in ll.split() ]
                r.append( ll[0] )
                dp.append( ll[1] )
        r=np.array(r); dp=np.array(dp)
        return (r,dp)


"""
Plot datas
"""

def plot(r, v, t=0):
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(
            r, v,
            # color = "orange",
            lw = 2
            )
        
        plt.grid(); 
        #plt.legend()
        if t!=0:
            ax.set_title("Scaled Potential with t=" + str('%.2f'%t))
        else:
            ax.set_title("Scaled Potential")
        ax.set_xlabel("Radius r")
        ax.set_xlim([0, 40])
        ax.set_ylim([-50, 5])
        ax.set_ylabel("Potential v([rho(r)],r)")
        
        
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
    
    energy = Energy(data=quickLoad("Densities/SkXDensityO16p.dat"), C_code=True, \
                    param_step=0.01, t_min=0.41, t_max=1.0, \
                    input_dir="Scaled_Potentials/O16", scaling='l')

    energy.solver()
    v = np.array(energy.vL)
    r = energy.R
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for i in range(len(v)):
        ax.plot(
            r, v[i,:],
            lw = 2
            )
    
    plt.grid(); #plt.legend()
    ax.set_title("Potentials for all lambdas")
    ax.set_xlabel("Radius r")
    ax.set_xlim([0, 12])
    # ax.set_ylim([-50, -100])
    ax.set_ylabel("Potential v")
    
    integrand = np.array(energy.integrand)
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for i in range(len(v)):
        ax.plot(
            r, integrand[i,:],
            lw = 2
            )
    
    plt.grid(); #plt.legend()
    ax.set_title("Integrand function for all lambdas")
    ax.set_xlabel("Radius r")
    ax.set_xlim([0, 12])
    # ax.set_ylim([-50, -100])
    ax.set_ylabel("Integrand values")
    
    """
    data = quickLoad("Scaled_Potentials/O16/Potentials/pot_L=0.410000_C++.dat", 3, 3)
    # data = quickLoad("Scaled_Potentials/O16/Potentials/pot_L=1.000000_C++.dat", 3, 3)
    # data_true = quickLoad("Potentials/pot_o16_skx_other_iks.dat")
    p = interpolate(data[0], data[1])
    
    v = scint.interp1d(data[0], data[1], fill_value="extrapolate")
    r = np.arange(0., data[0][-1], 0.1)
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(
        data[0], data[1],
        color = "blue",
        label = "original", 
        lw = 2
        )
    ax.plot(
        r, v(r),
        color = "green",
        ls = 'dashdot',
        label = "interp1D", 
        lw = 2
        )
    ax.plot(
        r, p(r),
        color = "orange",
        ls = '--',
        label = "spline", 
        lw = 2
        )
    """
    """
    ax.plot(
        data_true[0], data_true[1], # - data_true[1][-5],
        color = "Red",
        ls = 'dotted',
        label = "complete original", 
        lw = 2
        )
    """
    """
    plt.grid(); plt.legend()
    ax.set_title("Pot")
    ax.set_xlabel("Radius r")
    # ax.set_xlim([0, 11])
    # ax.set_ylim([-50, -100])
    ax.set_ylabel("Pot")
    """
    
    
    