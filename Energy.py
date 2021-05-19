import numpy as np
import os

from findiff import FinDiff
from scipy import integrate
import scipy.interpolate as scint
import matplotlib.pyplot as plt


from Orbitals import ShellModelBasis
from Problem import Problem
from Solver import Solver
from Misc import interpolate, floatCompare

"""

Methods
---------
getPotential
dEpot_dt
getPotentialEnergy

"""
class Energy(object):
    
    def __init__(self, problem=None, rho=None, v=None, grad_rho=None, data=[], \
                 param_step=0.01, t0=1., t=0., ts=None, \
                 R_min=1e-4, R_max=10., r_step=0.1, cutoff=1e-8, integrator=integrate.simpson,\
                 scaling="l", C_code=False, output="Output", input_dir="Scaled_Potentials", load=False):
        
        # Parameters
        if ts is not None:
            # t's provided as a list
            self.T = ts
            self.reverse_t = False
        else:
            self.T, self.reverse_t = self._setTParameters(t, t0, param_step)
            
        # Spatial mesh
        self.dr = r_step
        self.R = np.arange(R_min, R_max, self.dr)  
        if ( R_max-self.R[-1] )>1e-6 : 
            self.R = np.append(self.R, R_max)     
        self.d_dx  = FinDiff(0, self.dr, 1, acc=4)
        
        #saving output directory
        self.output = output
        #saving integration method
        self.integrator = integrator
        #saving scaling preference
        self.scaling = scaling
        
        # Using Problem
        if problem is not None:
            assert ( isinstance(problem, Problem) )
            self.problem = problem
            self.rho_grid = problem.grid
            self.tab_rho = problem.tab_rho  
            self.rho = problem.rho
            
            self.R = self.rho_grid
            self.tab_grad_rho = self.d_dx(self.tab_rho)
            self.grad_rho = interpolate(self.rho_grid, self.tab_grad_rho)
            
            self.method = "IKS python"
            self.load = load
            # density cutoff
            self.cutoff = cutoff
            self.blackList = []
        
        
        # using provided density
        elif (rho is not None or len(data)>0):
            assert( v is not None or C_code is True )
            
            # Input: density function
            if(rho is not None):
                self.rho = rho
                
                # gradient
                if (grad_rho is not None):
                    self.grad_rho = grad_rho
                else:
                    self.tab_rho = self.rho(self.R)
                    self.tab_grad_rho = self.d_dx(self.rho)
                    self.grad_rho = interpolate(self.R, self.tab_grad_rho)
            # Input: density array
            else:
                self.rho, self.grad_rho = interpolate(data[0], data[1], get_der=True)
            self.method = "Rho and v from input"
            
            # Potential 
            
            # Input potential function
            if(v is not None):
                self.v = v
            # Using datas from C++ code
            else:
                #saving method and file source directory
                self.input = input_dir
                self.method = "IKS C++"
            
            # create output directory
            if len(output)>0 and not os.path.exists("Results/" + output):
                    os.makedirs("Results/" + output)
                    
        else:
            assert(False), "Invalid input parameters"
            
        # Scaled densities: functions of r and t
        self.scaled_rho = scaleDensityFun(self.rho, scaling)
        self.drho_dt = self._getDrhoDt( scaling)
        
        # Dictionary of pot. functions
        self.dict_v = dict()
        # Kinetic energies
        self.kins = dict()
        
    
    def getEnergy(self):
        deltaU = self.getPotentialEnergy()
        deltaK = self.getKineticEnergyDiff()
        # TODO save somewhere
        return (deltaU+deltaK)
    
    
    " K[t]-K[t0] "    
    def getKineticEnergyDiff(self):
        # C++
        if self.method == "IKS C++":
            pass
        # Python
        else:
            deltaK = self.kins[ self.T[-1] ] - self.kins[ self.T[0] ]
            if self.reverse_t: deltaK = -deltaK
        return deltaK
        
            
        
    """
    Returns E[t]-E[t0]
    """
    def getPotentialEnergy(self):
        I = 0.
        de = []
        for t in self.T:
            de.append( self.dEpot_dt(t) )
        self.depot_dt = de
        I = self.integrator(de, self.T)
        if self.reverse_t: I = - I
        return I
        
        
    
    def dEpot_dt(self, t):
        r, v = self.getPotential(t)
        # integrand 
        f = r**2 * v * self.drho_dt(r,t)
        I = 4.*np.pi* self.integrator(f, r)
        return I
    
    
    """
    Calculate the potential for a given value of the scaling parameter t
    """
    def getPotential(self, t):
        # computing potentials with python IKS 
        if (self.method == "IKS python"):
            r, v, kin = self.IKS_Potential(t)
            # v function
            v_fun = interpolate(r, v)
            
        
        # computing potentials given from input
        elif(self.method == "Rho and v from input"):
            pass
        
        # loading potentials from C++ datas
        else:
            if self.scaling == "l" :
                r, v, v_fun = self.C_Potential(t)
                        
        self.dict_v[t] = v_fun
        return r, v
    
    
    def IKS_Potential(self, t):
        # scaled density
        rho_t = lambda r: self.scaled_rho(r, t)
        # setting problem's upper bound
        bound = getCutoff(rho_t, self.cutoff)
                
        print("\n\nComputing potential with ", self.scaling, \
              " scaling and parameter t: \t", t, '\n', \
              " upper bound: ", bound)
            
        # needed to check convergences
        check = (b"Algorithm terminated successfully at a locally optimal point, "
             b"satisfying the convergence tolerances (can be specified by options).")
        check2= b'Algorithm stopped at a point that was converged, not to "desired" tolerances, but to "acceptable" tolerances (see the acceptable-... options).'
        #setting output directory for each minimization
        file = self.output + "/" + self.scaling + "_minimization_t=" + str('%.4f'%t)
        
        # setting the scaled problem
        self.problem.setDensity(rho=rho_t, ub=bound, output_folder=file)
        r = self.problem.grid
        # computing eigenfunctions
        res, info = self.problem.solve() 
        kin = self.problem.kinetic
        
        if res['status'] != check and res['status'] != check2: 
            st = "Warning: convergence not reached for t=" + str('%.4f'%t) 
            st += "\nPotential set to zero."
            print (st)
            v = np.zeros_like(r)
            kin = 0.
        else:
            # computing potentials            
            solv = Solver(self.problem)         
            v = solv.getPotential()
            
        self.kins[t] = kin
        return r,v, kin
    
    
    
    """
    Load and calculate scaled potentials from C++ files
    """   
    def C_Potential(self, t):   
        #checking convergence status
        # self.checkCPotentials()     # TODO
        name = "/Potentials/pot_L=" + str('%.3f'%t) + "000_C++.dat"
        r, p = quickLoad(self.input + name, beg=3, end=3)
        # p = self.shiftPotentials(r, p)
        # creating a radial grid with the step given in input
        rad = np.arange(0., r[-1] + self.dr, self.dr)
        v_fun = scint.interp1d(r, p, fill_value="extrapolate")      ## with interp1d
        v = v_fun(rad)
        return r,v,v_fun
    
        

               
    def _getDrhoDt(self, scaling='l'):
        if scaling=='l':
            def drho_dt(r,t):
                return t**2* (3.*self.rho(t*r)+t*r*self.grad_rho(t*r) )
            return drho_dt
        # TODO add the other two scalings
        
        
        
    """
    Create list of parameters t
    """
    def _setTParameters(self, t, t0, step):
        self.dt = np.abs(step) 
        if t==0: t = self.dt
        if t0==0: t0 = self.dt
        
        if t > t0:
            reverse_t = False
            T = np.arange(t0, t, self.dt) 
            if (t-T[-1])>1e-6 : T = np.append(T, t)
        elif t0 == t:
            T = np.array([t0])
        elif t < t0: 
            reverse_t = True
            T = np.arange(t, t0, self.dt)
            if (t0-T[-1])>1e-6 : T = np.append(T, t0)
        return T, reverse_t
    
    
"""
Calculating density cutoff.
Return max. value of r such that rho(r)>cutoff
"""
def getCutoff(rho, cutoff=1e-8):
    for rr in np.arange(0.,50.,0.1):
        if( rho(rr) < cutoff ):
            return rr - 0.1
    
    
"""
Returns scaled density function (function of r and t)
"""    
def scaleDensityFun(rho, scale='l'):
    q = lambda r,t: t * rho(r)
    l = lambda r,t: t**3 * rho(t*r)
    z = lambda r,t: t**2 * rho(t**(1./3.)*r) 
    
    if scale == "q" :
        return q
    elif scale == "l" :
        return l
    elif scale == "z" :
        return z
    
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
    
    
def getTheoreticalPotentialScaled_t0t3(r, rho_p, rho_n, t):
    #parameters
    t0 = -2552.84; t3=16694.7; alfe=0.20309
    
    rho_p_fun_in = interpolate(r, rho_p)
    rho_n_fun_in = interpolate(r, rho_n)
    
    rho_p_fun = lambda r: scaleDensityFun(rho_p_fun_in, 'l')(r,t)
    rho_n_fun = lambda r: scaleDensityFun(rho_n_fun_in, 'l')(r,t)
    
    r_max = getCutoff(rho_n_fun)
    r = np.arange(0.,r_max+0.1,0.1)
    # """
    first = t0/4.+t3/24.*(rho_p_fun(r)+rho_n_fun(r))**alfe
    second = t3/24.*alfe*(rho_p_fun(r)+rho_n_fun(r))**(alfe-1)
    third = (rho_p_fun(r)+rho_n_fun(r))**2 + 2*rho_p_fun(r)*rho_n_fun(r)
    
    vp = first * (2*rho_p_fun(r)+4*rho_n_fun(r)) + second * third
    vn = first * (2*rho_n_fun(r)+4*rho_p_fun(r)) + second * third
    """
    vp = 3./2.*t0*rho_p_fun(r)+\
        t3/4.*(2*rho_p_fun(r))**alfe*rho_p_fun(r)+\
            t3/4.*alfe*(2*rho_p_fun(r))**(alfe-1)*rho_p_fun(r)**2
    vn = vp
    """
    ### OK, they are the same as long as the scaling is applied to both densities
    return r, vp, vn
    


"""
TODO
- use tabulate density instead of function in IKS_Potential?
- check it works correctly and fast for t=1
- have we tested setDensity?
"""
    
if __name__ == "__main__":
    
    data = quickLoad("Densities/rho_o16_t0t3.dat")
    rho = interpolate(data[0], data[1])
    
    lam = 0.90
    rho_l = lambda r: scaleDensityFun(rho)(r,lam)
    
    plt.figure(0)
    plt.plot(data[0], rho(data[0]), label="l=1")
    plt.plot(data[0], rho_l(data[0]), label="l={}".format(lam))
    plt.legend()
    plt.grid()
    plt.xlim(0.,6.)
    
    """
    r,v = quickLoad("Potentials/pot_o16_t0t3.dat")
    for lam in np.arange(0.8,1.2,0.1):
        plt.figure(2)
        r, vp, vn= getTheoreticalPotentialScaled_t0t3(data[0], data[1], data[1], lam)
        plt.plot(r, vp, label="{:.2f}".format(lam) )
        plt.legend()
    """   
    
    plt.figure(3)
    r,v = quickLoad("Potentials/pot_o16_t0t3.dat")
    v = interpolate(r,v)
    
    dummy = Problem(8,8, data=data, ub=8., n_type='p')
    results, info = dummy.solve()
    solver = Solver(dummy)
    pot = solver.getPotential()
    
    y = pot-v(dummy.grid)
    plt.plot(dummy.grid, y-np.max(y) )
    plt.xlim(0.,6.)
    
    
    
    
    """
    #dummy = Problem(8,8, data=data, ub=8., n_type='p')
    dummy = Problem(8,8, rho=rho_l, ub=10., n_type='p')
    
    results, info = dummy.solve()
    solver = Solver(dummy)
    
    
    
    
    plt.figure(1)
    pot = solver.getPotential()
    plt.plot(solver.grid, pot- pot[-20]+vp[-20], '--', label="CV")
    plt.plot(r, vp, label="t0t3")
    plt.xlim(0.,10.); plt.ylim(-70.,25.)
    plt.grid(); plt.legend()
    
    #energy = Energy(problem=dummy,data=data,cutoff=1e-8)
    #r,v = energy.getPotential(1.)
    """