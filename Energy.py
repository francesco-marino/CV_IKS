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

from Plot_templ import Plot
from Orbitals import ShellModelBasis
from Problem import Problem
from Solver import Solver

"""
TO DO/CHECK LIST: 
    
--- NB: IKS potentials nearby r=0 are not trustworthy

--- check where rho==0 (see line 74) (ex HO_20_particles_coupled_basis), it gives problem somewhere

--- split computation of Q/Lambda/Z path? better readable 
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
        step for the parametric intergal (default: 0.01)
    r_step : float 
        step for the radial integral (default: 0.1)
    R_min : float
        lower bound of the radial integral (default: 1e-4)
    R_max : float
        upper bound of the radial integral (default: 10.)
    
    """
    
    def __init__(self, problem=None, rho=None, v=None, grad_rho=None, data=[],\
                 output="Output", param_step=0.01, t_min=1., t_max=0., ts=None, r_step=0.1,\
                 R_min=1e-4, R_max=10., cutoff=1e-8, integrator=integrate.simpson,\
                 scaling="l", C_code=False, input_dir="Scaled_Potentials", load='y'):
        
        #NOTES:
            #self.R must be chosen in order to consider all potentials after they have gone to 0. (IKS)
            
        self.dr = r_step
        self.R = np.arange(R_min, R_max, self.dr)  #radius value list ##1
        if (R_max-self.R[-1])>1e-6 : self.R = np.append(self.R, R_max)
        
        """
        num = int((R_max - R_min) / self.dr)
        self.R = np.linspace(R_min, R_max, num)  #radius value list ##2
        # real_r_step = self.R / num
        self.dr = (self.R[-1]-self.R[0])/num
        """
        
        self.d_dx  = FinDiff(0, self.dr, 1, acc=4)
            
        #creating parameter list
        self.dt = param_step
        self.reverse_t = False
        if ts is not None: 
            self.T = ts
        else: 
            self.setNewParameters(t_min, t_max, param_step)
        
        
        #if more methods are present, IKS problem comes first
        
        #IKS problem
        if (problem is not None):
            
            
            #------------ADD control over rho==0!! (np.any / np.all)
            self.rho_grid = problem.grid
            self.rho = problem.tab_rho
            self.rho_fun = problem.rho
            #print(self.rho_grid, self.rho)
            
            self.R = self.rho_grid
            self.grad_rho = self.d_dx(self.rho)
            self.grad_rho_fun = interpolate(self.rho_grid, self.grad_rho)
            # self.rho_fun, self.grad_rho_fun = interpolate(self.rho_grid, self.rho, der=True)

            self.cutoff = cutoff
            
            self.blackList = []
            
            #saving problem and method
            self.problem = problem
            self.method = "IKS python"
            self.load = load
            
        # using provided density
        elif(rho is not None or len(data)>0): 
            assert( v is not None or C_code is True ) 
            
            # Input density function
            if(rho!=None):
                self.rho_fun = rho
                
                # provided density gradient
                if (grad_rho!=None):
                    self.grad_rho_fun = grad_rho
                    # self.grad_rho = grad_rho(self.R)
                
                # numerical gradient (if none is given)
                else:
                    self.rho = self.rho_fun(self.R)
                    self.grad_rho = self.d_dx(self.rho)
                    self.grad_rho_fun = interpolate(self.R, self.grad_rho)
            
            # Input density array
            else:
                self.rho_fun, self.grad_rho_fun = interpolate(data[0], data[1], der=True)
                # self.rho = self.rho_fun(self.R)
                # self.grad_rho = self.grad_rho_fun(self.R)
                
            self.method = "Rho and v from input"
            
            # Input potential function
            if(v!=None):
                self.v_fun = v
                
            # Using datas from C++ code
            else:
                #saving method and file source directory
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
        self.N = 4*np.pi*self.integrator(self.R**2 * self.rho_fun(self.R), self.R)
        
        
    """
    Compute total energy (Kinetic + Potential)
    """    
    def getEnergy(self):
        # potential
        DeltaU = self.getPotential_En()
        # kinetic
        K = self.getKinetic_En()
        DeltaK = K[1]-K[0]
        # total
        DeltaE = DeltaU + DeltaK
        
        # Save results to file
        self._saveE(DeltaU, "Potential energy")
        self._saveE(DeltaK, "Kinetic energies")
        self._saveE(DeltaE, "Total energies")
        
        return DeltaE
        
        
    """
    Return system energy w/ Q, K, L scaling
    
    Returns
    --------
    E : float []
        energy array from all scalings
        
    """
    
    def getPotential_En(self):
        
        # computing potentials with python IKS 
        if (self.method == "IKS python"):
            
            rQ, vQ, rL, vL, rZ, vZ = self.IKS_Potential()
            
            #they are useful only for q, not anywhere else (and only self.rho too)
            #getting density and its gradient in self.R ###remove maybe??
            # self.rho = self.rho_fun(self.R)
            # self.grad_rho = self.d_dx(self.rho)
            
            if self.scaling == "all" or self.scaling == "q" :
                self.rQ, self.vQ = self.P_Potential(rQ, vQ, self.T_Q)
                
            if self.scaling == "all" or self.scaling == "l" :
                self.rL, self.vL = self.P_Potential(rL, vL, self.T_L)
                
            if self.scaling == "all" or self.scaling == "z" :
                self.rZ, self.vZ = self.P_Potential(rZ, vZ, self.T_Z)
        
        # computing potentials given from input
        elif(self.method == "Rho and v from input"):
            
            self.vQ, self.vL, self.vZ = self.Input_Potential()
            
            self.rQ = np.ones_like(self.vQ) * self.R; self.T_Q = self.T
            self.rL = np.ones_like(self.vL) * self.R; self.T_L = self.T
            self.rZ = np.ones_like(self.vZ) * self.R; self.T_Z = self.T
        
        # loading potentials from C++ datas
        else:
            
            if self.scaling == "all" or self.scaling == "l" :
                self.rL, self.vL = self.C_Potential()
                self.T_L = self.T
            
            
        #print("v Q",self.vQ, "\n\n L", self.vL, "\n\n Z",\
            #self.vZ, "\n\n Drho", self.grad_rho)
        
        # Compute the full (parametric + spatial) integral
        U = self.calcIntegral()
    
        return U[1]
        
        
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
        
        # create folder in which to save potentials and kinetic energies
        if not os.path.exists("Results/" + self.output + "/Potentials"):
                    os.makedirs("Results/" + self.output + "/Potentials")
        if not os.path.exists("Results/" + self.output + "/Kinetics"):
                    os.makedirs("Results/" + self.output + "/Kinetics")
                    
        # needed to check convergences
        check = (b"Algorithm terminated successfully at a locally optimal point, "
             b"satisfying the convergence tolerances (can be specified by options).")
        
        if self.scaling == "all" :
            scaling = ["q", "l", "z"]
        else:
            scaling = self.scaling
            
        vQ=[]; vL=[]; vZ=[]
        rQ=[]; rL=[]; rZ=[]
        
        for s in scaling:
            r=[]; v=[]; K=[]
            status=[]; elim=[]
            
            for j,t in enumerate(self.T): 
                
                # t=round(t,2)
                
                # output file for potentials
                file_pot = "Results/" + self.output + \
                    "/Potentials/pot_t=" + str('%.4f'%t) + ".dat"
                    
                # output file for kinetic energy
                file_kin = "Results/" + self.output + \
                    "/Kinetics/k_t=" + str('%.4f'%t) + ".dat"
                    
                # if potential and kinetic are already calculated and load == 'y', then load them
                if os.path.isfile(file_pot) and os.path.isfile(file_kin) \
                    and self.load=='y':
                    radius, potential = quickLoad(file_pot)
                    r.append(radius)
                    v.append(potential)
                    
                    with open(file_kin) as f:
                        K.append( float(f.readlines()[0]) )
                    continue
                
                # check if t is a bad parameter
                elif floatCompare(t, self.blackList):
                    elim.append(j)
                    continue
                
                # getting scaled density function 
                Rho_fun = scaleDensityFun(self.rho_fun, t, s)

                # setting problem's upper bound
                bound = self.getCutoff(Rho_fun)
                
                print("\n\nComputing potential with ", s, \
                      " scaling and parameter t: \t", t, '\n', \
                      " upper bound: ", bound)
                
                #setting output directory for each minimization
                file = self.output + "/" + s + "_minimization_t=" + str('%.4f'%t)
                
                # setting the scaled problem
                self.problem.setDensity(rho=Rho_fun, ub=bound, output_folder=file)
                
                # computing eigenfunctions
                res, info = self.problem.solve()
                
                # saving status regardless of the convergence success 
                st = "Minimization with t = "+ str('%.4f'%t) + " : " + str(res['status'])
                status.append(st)
                
                #checking convergence
                if res['status']!=check : 
                    self.blackList.append(t)
                    elim.append(j)
                    continue
                
                # computing potentials
                solv = Solver(self.problem)
                v.append(solv.getPotential())
                r.append(solv.grid)
                
                # saving potentials on a separated folder
                save = np.column_stack((r[-1],v[-1]))
                np.savetxt(file_pot, save)
                
                # Plot(r[-1], v[-1], "pot t="+str('%.4f'%t))
                
                # saving kinetic energy on a separated folder
                K.append(self.problem.kinetic)
                np.savetxt(file_kin, np.array([K[-1]]))
                
            #saving status
            self.saveStatus(status, s)
            
            #saving potentials and modifying parameter list
            if s == "q":
                self.T_Q = np.delete(self.T, elim)
                # self.status_Q = status
                self.K_Q = K
                rQ = r
                vQ = v
            elif s == "l":
                self.T_L = np.delete(self.T, elim)
                # self.status_L = status
                self.K_L = K
                rL = r
                vL = v
            elif s == "z":
                self.T_Z = np.delete(self.T, elim)
                # self.status_Z = status
                self.K_Z = K
                rZ = r
                vZ = v
                
        # they are all evaluated in different r (due to cutoff)
        return rQ, vQ, rL, vL, rZ, vZ
       
        
    """
    Calculating density cutoff
    """
    
    def getCutoff(self, rho):
        """
        if c==0.91 or c==0.98:
            r=0.
            while(rho(r) > self.cutoff):
                r += 0.1
            
            return r - 0.1
        else:
        """
        for rr in np.arange(0.,50.,0.1):
            if( rho(rr) < self.cutoff ):
                return rr - 0.1
                     
            
    """
    Computing parametric potential given from input
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
            vL.append(row)
            # Z
            x = self.R * t**(1./3.)
            row = self.v_fun(x, self.rho_fun, self.N, t**2, self.R)
            vZ.append(row)
        
        return np.array(vQ), np.array(vL), np.array(vZ)
    
    
    """
    Load and adjust python potentials 
    """
    
    def P_Potential(self, r, v, T):
    
        vT=[]; rT=[]
        for i, t in enumerate(T):

            # removing last 10 points
            elim = np.arange(-10,0,1)
            r[i] = np.delete(r[i], elim)
            v[i] = np.delete(v[i], elim)
            
            v[i] = self.shiftPotentials(r[i], v[i])
            
            # Plot(r[i], v[i], "pot t="+str('%.4f'%t)+" shifted")
            
            # creating a radial grid with the step given in input
            rad = np.arange(0., r[i][-1] + self.dr, self.dr)
            rT.append( rad )
            
            # interpolation
            v_fun = interpolate(r[i], v[i])                                   ## with spline
            # v_fun = scint.interp1d(r[i], v[i], fill_value="extrapolate")    ## with interp1d
            
            vT.append( v_fun( rT[i] ) )
            
            # Plot(r[i], vT[i], "pot t="+str('%.4f'%t)+" shifted and interpoled")
        
        return rT, vT
    
    
    """
    Load and calculate scaled potentials from C++ files
    """
    
    def C_Potential(self):
        #evaluating each in different r, as in P_Potentials

        #checking convergence status
        self.checkCPotentials()
    
        rT=[]; vT=[] 
        for i, t in enumerate(self.T):
            name = "/Potentials/pot_L=" + str('%.3f'%t) + "000_C++.dat"
            r, p = quickLoad(self.input + name, beg=3, end=12)
            p = self.shiftPotentials(r, p)
            
            # creating a radial grid with the step given in input
            rad = np.arange(0., r[-1] + self.dr, self.dr)
            rT.append( rad )
            """
            num = int(r[-1] / self.dr)
            rT.append(np.linspace(0., r[-1], num))
            """
            
            # interpolation
            # v_fun = interpolate(r, p)                                 ## with spline
            v_fun = scint.interp1d(r, p, fill_value="extrapolate")      ## with interp1d
            vT.append(v_fun(rT[i]))
            
            # plot(rT[i], vT[i], t)
        
        return rT, vT
    
    
    """
    Loading kinetics energies from C++ files
    """
    
    def C_Kinetics(self):
        K=[]
        for i, t in enumerate(self.T):
            name = "/Kinetics/kin_L=" + str('%.3f'%t) + "000_C++.dat"
            with open(self.input + name) as f:
                K.append( float( f.readlines()[0] ) )
        
        return K
    
    
    """
    Create parametrical grid while checking convergence of C++ potentials
    """
    
    def checkCPotentials(self):
        b, l = quickLoad(self.input + "/Status.dat")
            
        elim = np.where(b!=1)
        for e in elim[0]:
            if floatCompare(l[e], self.T):
                rem = np.where(abs(l[e]-self.T)<1e-6)
                self.T = np.delete(self.T, rem)
        
        
    """
    Shifting potentials so that v(r)=0 for r->inf
    """
    ######## NOT USED AT THE MOMENT (maybe used but it does nothing) ##########
    def shiftPotentials(self, rad, pot):
        # DeltaPot = np.amax(pot) - np.amin(pot)
        pot = pot #- pot[-5]
        
        return pot
    ##########################################    
    
    """
    Extend potentials with 0 till the radial grid end
    """
    ######## NOT USED AT THE MOMENT ##########
    def extendPotentials(self, rad, pot):
        R = rad[-1]
        while (R < self.R[-1]):
            # print(t)
            R += 0.1
            rad = np.append(rad, R)
            pot = np.append(pot, 0.)
            
        return rad, pot
    ###########################################
    
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
            
            I_r=[]
            for j, l in enumerate(self.T_Q):
                # Q
                x = self.rQ[j]
                f_Q = self.vQ[j] * 4 * np.pi * x**2 * self.rho_fun(x) / self.N 
                #integrating over r
                I = self.integrator(f_Q, x)
                I_r.append(I)
                
            # saving integrand function   
            self.dE_dQ = I_r
            
            #integrating over t
            I_Q = self.integrator(I_r, self.T_Q)
            
            if self.reverse_t == True: I_Q = - I_Q


        if self.scaling == "all" or self.scaling == "l" :
            #LAMBDA integral
            
            I_r=[]
            self.integrand=[]
            for j, l in enumerate(self.T_L):
                # L
                x = self.rL[j] * l
                f_L = self.vL[j] / l * 4 * np.pi * x**2 * \
                    (3 * self.rho_fun(x) / self.N + x * self.grad_rho_fun(x) / self.N)
                #integrating over r
                I = self.integrator(f_L, x)
                I_r.append(I)
                
                #graphic purposes
                self.integrand.append(f_L)
            
            
            self.integral_L = I_r
            # saving integrand function
            self.dE_dL = I_r
            
            #integrating over t
            I_L = self.integrator(I_r, self.T_L) 
            
            if self.reverse_t == True: I_L = - I_L

        
        if self.scaling == "all" or self.scaling == "z" :
            #Z integral
            
            I_r=[]
            for j, z in enumerate(self.T_Z):
                # Z
                x = self.rZ[j] * z**(1./3.) 
                f_Z = self.vZ[j] * 4 * np.pi * x**2 * \
                    (2 * self.rho_fun(x) / self.N + x/3. * self.grad_rho_fun(x) / self.N)
                #integrating over r
                I = self.integrator(f_Z, x)
                I_r.append(I)
            
            self.dE_dZ = I_r
            #integrating over t
            I_Z = self.integrator(I_r, self.T_Z)
            
            if self.reverse_t == True: I_Z = - I_Z
        
        return I_Q, I_L, I_Z
    
    
    """
    Returning integrand of parameter integrals (energy derivatives)
    """
    
    def dEdt(self):
        
        if self.scaling == "all" :
            return self.T_Q, self.dE_dQ,\
                    self.T_L, self.dE_dL,\
                     self.T_Z, self.dE_dZ
        if self.scaling == "q" :
            return self.T_Q, self.dE_dQ
        if self.scaling == "l" :
            return self.T_L, self.dE_dL
        if self.scaling == "z" :
            return self.T_Z, self.dE_dZ
    
    
    """
    Setting new parameters list (only for IKS C++ method)
    """
    
    def setNewParameters(self, t0=1.0, t=1.0, step=0):
        if step>0: self.dt = step 
        if t==0: t = self.dt
        if t0==0: t0 = self.dt
        
        if t > t0:
            self.reverse_t = False
            self.T = np.arange(t0, t, self.dt) 
            if (t-self.T[-1])>1e-6 : self.T = np.append(self.T, t)
        elif t0 == t:
            self.T = np.array([t0])
        elif t < t0: 
            self.reverse_t = True
            self.T = np.arange(t, t0, self.dt)
            if (t0-self.T[-1])>1e-6 : self.T = np.append(self.T, t0)
        # print(self.T)
     
    
    """
    Return kinetic energy and its parameters
    """
    
    def getKinetic_En(self):
        # C++
        if self.method == "IKS C++":
            self.K_L = self.C_Kinetics()
            return self.T_L, self.K_L
        # Python
        else:
            if self.scaling == "all" :
                return self.T_Q, self.K_Q,\
                        self.T_L, self.K_L,\
                         self.T_Z, self.K_Z,
            if self.scaling == "q" :
                return self.T_Q, self.K_Q
            if self.scaling == "l" :
                return self.T_L, self.K_L
            if self.scaling == "z" :
                return self.T_Z, self.K_Z
                  
        
 
    
    """
    Interpolate and evaluate C0 C1 C2 in the same grid as f 
    """
    def getCFunctions(self, r):
        C0, C1, C2 = self.problem._getCFunctions()
        C_0 = []
        for i in range(len(C0)):
            fun = interpolate( self.problem.grid, C0[i] )
            C_0.append( fun(r) )
        C = []
        for c in [C1, C2]:
            fun = interpolate( self.problem.grid, c )
            C.append( fun(r) )
            
        return C_0, C[0], C[1]
        
        
        
        
    """
    Printing Potential energies on file
    
    Parameters
    ----------
    E : float []
        energy array from all scalings
        
    """
    
    def _saveU(self, E):
        self.E_file = "Results/" + self.output + "/E.out"
        with open(self.E_file, 'w') as fo:
            fo.write("\n \n Potential energies: \n Q-path:\t" + str(E[0]) + "\n\n")
            fo.write("\n Lambda-path:\t" + str(E[1]) + "\n\n")
            fo.write("\n Z-path:\t" + str(E[2]) + "\n\n")
    
    
    """
    Printing Potential energies on file
    
    Parameters
    ----------
    E : float
        Energy to save
    label : string
        Energy label
        
    """
    
    def _saveE(self, E, label):
        self.E_file = "Results/" + self.output + "/E.out"
        with open(self.E_file, 'w') as fo:
            fo.write("\n \n"+label+" : "+str(E) + "\n\n")
    
    
    """
    Saving status on file
    """
    def saveStatus(self, status, scale):
        save = np.reshape(status, newshape=(-1,1))
        file_stat = "Results/" + self.output + "/Status_"+scale+".dat"
        np.savetxt(file_stat, save, delimiter='  ', \
                   header='string', comments='', fmt='%s')
            
            
    """
    Returnig the potential
    """
    
    def _getPot(self):
        return self.R, self.vQ, self.vL, self.vZ
    

"""
Calculating scaled density function
"""

def scaleDensityFun(rho, t, scale):
    q = lambda r: t * rho(r)
    l = lambda r: t**3 * rho(t*r)
    z = lambda r: t**2 * rho(t**(1./3.)*r) 
    
    if scale == "all" :
        return q, l, z
    if scale == "q" :
        return q
    if scale == "l" :
        return l
    if scale == "z" :
        return z
    
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
Read files like u.dat and f.dat
"""

def loadUF(file):
    
    file = open(file)
    ff = file.readlines()
    file.close()
    R = []; U = []; orb = []
    for j,ll in enumerate(ff):
        if (str(ll).startswith("#") and j==0) :
            orb.append(ll)
            r = []; u = []
        elif ( str(ll).startswith("#") ):
            orb.append(ll)
            R.append(r)
            U.append(u)
            r = []; u = []
        else:
            ll = [ float(x) for x in ll.split() ]
            r.append( ll[0] )
            u.append( ll[1] )

    R.append(r)
    U.append(u)     
    R=np.array(R); U=np.array(U)
    return (R,U,orb)    


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
        # ax.set_xlim([0, 40])
        # ax.set_ylim([-50, 5])
        ax.set_ylabel("Potential v([rho(r)],r)")
        
        
def floatCompare(a, floats, **kwargs):
  return np.any(np.isclose(a, floats, **kwargs))
      
#########################################################
########################TEST#############################
#########################################################
from E_behaviour import E_behaviour_calc

if __name__ == "__main__":
     
    #check which cutoff was used in C++ simulations (check directory name)
    prec = "_-8"
    
    #choose which nucleus, particle and interaction to use:
    # n = 1 O16n SkX
    # n = 2.5 O16n t0t3
    # n = 8 Ca40n Skx
    # n = 9 Ca40n t0t3
    # see E_behaviour for the full list  (for other n the program should be slightly modified usually)
    n = 1
    
    #max iteration for the IKS code, if all the potentials have already been calculated \
    #(thus saved in the directory), it can be set to 0
    max_iter = 2000
    
    E_behaviour_calc(prec, n, max_iter)
    
        