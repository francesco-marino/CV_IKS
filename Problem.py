# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:28:43 2020

@author: Francesco
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import ipopt
from findiff import FinDiff

from Orbitals import ShellModelBasis,  OrbitalSet, getOrbitalSet
from Constants import T, nuclearOmega, nuclearNu
from Misc import simpsonCoeff, saveData, loadData
from EigenFunctions import HO_3D




class Problem(ipopt.problem):
    """
    A class used to represent a nuclear Inverse Kohn-Sham (IKS) problem.
    
    
    Outline of the IKS callbacks:
        objective:
            the function to be minimized (the kinetic energy)
        gradient:
            gradient of the objective function
        constraints:
            constraint functions [density and overlap integral (orthonormality)] 
        jacobianstructure:
            non-zero elements of the jacobian (sparse matrix)
        jacobian:
            jacobian matrix of the constraints
        hessianstructure (TODO)
        hessian (TODO)

    ...

    Parameters
    ----------
    Z : int
        Number of protons
    N : int
        Number of neutrons (default 0)
    rho : function
        target density function (default None)
    lb : float
        lower bound of the mesh (default 0.1)
    ub : float
        upper bound of the mesh (default 10.)
    h : float
        step (default 0.1)
    n_type : string
        run calculations for either protons ('p') or neutrons ('n') (default 'p')
    data : list
        if rho is None, generate target density by interpolating data[0] (r) and data[1] (rho) with a spline (default [])
    basis : orbital.OrbitalSet
        basis, described by quantum numbers nlj or nl (default orbital.ShellModelBasis, i.e. nlj)
    max_iter : int
    rel_tol : float
    constr_viol : float
    output_folder : str
        name of the folder inside Results where the output is saved (default Output)
    debug : str
        (not implemnted yet)
        
    
    Methods
    -------
    
    """
    
    def __init__(self,Z,N=0,rho=None, lb=0.1,ub=10., h=0.1, n_type="p", data=[], basis=ShellModelBasis(),\
        max_iter=2000, rel_tol=1e-3, constr_viol=1e-3, output_folder="Output", debug='n'):
        
        # Basic info.
        self.N = N
        self.Z = Z
        # "n" or "p"
        self.n_type = n_type if (n_type=="p"or n_type=="n") else "p"
        # Either N or Z
        self.n_particles = Z if self.n_type=="p" else N
        
        # Box
        self.lb = lb
        self.ub = ub
        self.h  = h
        self.n_points = int( (ub-lb)/h ) + 1
        # Spatial grid [lb, lb+h, ..., ub]
        self.grid = np.linspace(lb, ub, self.n_points)
        # Integration factors
        # self.h_i  = np.array([simpsonCoeff(i,self.n_points) for i in range(self.n_points) ]) * self.h/3.
        
        # Derivative operators
        self.d_dx  = FinDiff(0, self.h, 1, acc=4)
        self.d_d2x = FinDiff(0, self.h, 2, acc=4)
         
        # Orbitals
        self.basis = basis
        self.orbital_set = getOrbitalSet(self.n_particles, basis)
        self.n_orbitals = len(self.orbital_set)
        # Pairs (i,j) of non-orthogonal orbitals
        self.pairs = self.getPairs()
        
        # Nu & omega (for starting wave functions)
        self.nu = nuclearNu(self.n_particles)
        self.omega = nuclearOmega(self.n_particles)
        
        # Total n. variables 
        self.n_variables = self.n_orbitals*self.n_points
        # N. constraints (density + orthonormality)
        self.n_constr = self.n_points + len(self.pairs)
        
        # Density and its derivatives
        self.data = data
        self.rho = rho if rho!=None else self.getRhoFromData( data[0], data[1] )
        # Tabulate rho and its derivatives
        self.tab_rho = self.rho(self.grid)
        self.d1_rho  = self.d_dx( self.tab_rho)
        self.d2_rho  = self.d_d2x(self.tab_rho)   
        # Integration factor: 4 pi r^2 rho
        self.rho_r2  = 4.*np.pi * self.tab_rho * np.power(self.grid, 2)       
        # C0,C1,C2
        self.C0, self.C1, self.C2 = self._getCFunctions()
        
        
        # Create the output directory
        self.output_folder = "Results/"+output_folder if len(output_folder)>0 else ""
        if len(self.output_folder)>0 and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.debug = True if debug=='y' else False
        #self.dbg_file = open(self.output_folder+"/debug.dat", 'w')
        
        # Output files
        # Rescaled orbitals f(r) 
        self.f_file = self.output_folder + "/f.dat"
        # Radial orbitals u(r)
        self.u_file = self.output_folder + "/u.dat"
        self.pot_file = self.output_folder + "/potential.dat"
        self.epsilon_file  = self.output_folder + "/epsilon.dat"
        
        # Dictionary of results
        self.results  = dict()
        self.datafile = self.output_folder + "/data"
        
        # Initialize ipopt
        self._setUp(max_iter,rel_tol,constr_viol)
        
        
    
    
    """
    Change the target  density 
    
    Parameters
    ----------
    rho:
        density function (same as __init__)
    data: 
        data (r, rho(r) ) (same as __init__)
    output_folder: str
        name of the output folder inside Results
    
    """
    def setDensity(self, rho=None, data=[], output_folder="Output"):
        if rho is None and len(data)==0:
            raise ValueError("Error: no valid density was provided")
        else:
            self.basis.reset()
            self.__init__(self.Z, self.N, rho, self.lb, self.ub, self.h, self.n_type, data, \
               self.basis, self.max_iter, self.rel_tol, self.constr_viol, output_folder, self.debug )
    
    
    """
    Objective function
    O[f] = (-T) sum_j( d_j I_j )
    
    Parameters
    ----------
    x: np.array(self.n_variables)
        current value of the rescaled orbitals f_j
    
    Returns
    ----------
    obj: float
        the current value of the objective function
    
    
    """
    def objective(self, x):
        # Reshape x into matrix and get its derivatives
        x = np.reshape(x, (self.n_orbitals,self.n_points) )
        d1x, d2x = self._deriv(x)
        # Sum_j (d_j integral_j)
        arr = np.array([ self.orbital_set[j].occupation * self._integral_j(x, d1x, d2x, j) for j in range(self.n_orbitals) ])
        obj = (-T) * np.sum(arr)
        return obj
    
    
    """
    Returns I_j. Called in objective
    """
    def _integral_j(self, x, d1x, d2x, j):
        arr = x[j,:] * ( self.C0[j,:]*x[j,:] + self.C1*d1x[j,:] + self.C2*d2x[j,:] )
        # arr = self.C0[j,:] * x[j,:]**2  - self.C2 * d1x[j,:]**2
        return np.sum( arr*self.h )
    
    
        
    """
    Gradient of the objective function dO/d[f_q(p)]
    
    Returns
    ----------
    grad: np.array(self.n_variables)
        gradient (1d array)
    """
    def gradient(self, x):
        x = np.reshape(x, (self.n_orbitals,self.n_points) )
        grad = np.zeros_like(x)
        d1x, d2x = self._deriv(x)
        for q in range(self.n_orbitals):
            deg = self.orbital_set[q].occupation
            rhs = 2.*( self.C0[q,:]*x[q,:] + self.C1*d1x[q,:] + self.C2*d2x[q,:] )  
            # HERE
            grad[q,:] = (-T)* deg * rhs * self.h
        # Reshape into 1d array
        return np.ndarray.flatten(grad)
    
    
    
    """
    Current value of the constraint functions
    
    Returns
    ----------
        : np.array(self.n_constraints)
        constraints at a given x
    """        
    def constraints(self, x):
        x = np.reshape(x, (self.n_orbitals,self.n_points) )
        # Density constr.
        dens = np.sum( [self.orbital_set[j].occupation * x[j,:]**2 for j in range(self.n_orbitals)], axis=0)
        # Ortho.
        ortho = np.zeros( len(self.pairs) )
        # <i|j> = int( 4 pi r^2 rho(r) f_i(r) f_j(r) )
        for k, (i,j) in enumerate(self.pairs):
            f_i = x[i,:]; f_j = x[j, :]
            ortho[k] = np.sum( self.rho_r2 * f_i*f_j ) *self.h 
        return np.concatenate( (dens,ortho) )
        
    
    
    """
    Returns (rows, cols) of non-zero elements of the jacobian of the constraints
    """
    def jacobianstructure(self):
        # Jac -> (constr, orbital, point)
        jac=np.zeros( (self.n_constr,self.n_orbitals,self.n_points) )
        # dg(r)/df_q(r')
        for q in range(self.n_orbitals):
            np.fill_diagonal( jac[:self.n_points, q, :], np.ones(self.n_points) )   # trick to enforce r=r'
        # Orthonormality constraints
        for k, (i,j) in enumerate(self.pairs):
            counter = self.n_points + k
            # dG_ij/df_i(r) for each r
            jac[counter, i, :] += np.ones(self.n_points) 
            # dG_ij/df_j(r) for each r
            jac[counter, j, :] += np.ones(self.n_points)
        # Shape into 2-matrix form
        jac = jac.reshape( (self.n_constr,self.n_variables) )
        return np.nonzero( jac )
    
    """
    Jacobian of the constraints
    """
    def jacobian(self, x):
        x = np.reshape(x, (self.n_orbitals,self.n_points) )
        # Jac -> (constr, orbital, point)
        jac = np.zeros( (self.n_constr,self.n_orbitals,self.n_points) )
        # Density
        for q in range(self.n_orbitals):
            deg = self.orbital_set[q].occupation
            np.fill_diagonal( jac[:self.n_points, q, :],  2.* deg * x[q,:] )  
        # Orthonormality
        for k, (i,j) in enumerate(self.pairs):
            counter = self.n_points + k
            # dG_ij/df_i(r) for each r    -> J!!! on the rhs   
            jac[counter, i, :] += self.rho_r2 * x[j,:] * self.h
            # dG_ij/df_j(r) for each r    -> I!!! on the rhs
            jac[counter, j, :] += self.rho_r2 * x[i,:] * self.h
        # Reshape into 2-matrix
        jac = jac.reshape( (self.n_constr,self.n_variables) )
        # Get non-zero elements
        return jac[ self.jacobianstructure() ]
        
            
    
    
    def hessianstructure(self):
        # Hessian -> (q, p, q', p')
        hess = np.zeros( (self.n_orbitals,self.n_points, self.n_orbitals,self.n_points) )
        # Obj. and density constr. contributions ( q = q' )
        for q in range(self.n_orbitals):
            np.fill_diagonal( hess[q,:,q,:], np.ones(self.n_points) )
        # Orthogonality
        for (i,j) in self.pairs:
            if i!=j:            # Add off-diagonal ortho. terms
                np.fill_diagonal( hess[i,:,j,:], np.ones(self.n_points) )
        # Reshape into 2-matrix
        hess = np.reshape( hess, (self.n_variables,self.n_variables) )
        return np.nonzero( hess )
        
    
    
    def hessian(self, x, lagrange, obj_factor):
        x = np.reshape(x, (self.n_orbitals,self.n_points) )
        # Hessian -> [q, p, q', p']  (The Hessian is diagonal: p=p' always)
        hess = np.zeros( (self.n_orbitals,self.n_points, self.n_orbitals,self.n_points) )
        # Obj. and density constr. contributions ( q = q' )
        for q in range(self.n_orbitals):
            deg = self.orbital_set[q].occupation
            # Objective function    HERE
            # rhs  = obj_factor * 2.*deg *(-T) * ( self.C0[q,:] -self.C1 + self.C2 ) *self.h
            rhs  = obj_factor * 2.*deg *(-T) * self.C0[q,:] *self.h
            # Density constraint
            rhs += lagrange[:self.n_points] * 2.*deg
            np.fill_diagonal( hess[q,:,q,:], rhs )
        # Orthonormality constraints
        for k, (i,j) in enumerate(self.pairs):
            delta = 2. if i==j else 1.
            # q=i, q'=j  (i>=j)  (No need to insert q=j,q'=i)      HERE
            rhs = lagrange[self.n_points+k] *delta * self.rho_r2 * self.h
            hess[i,:,j,:] += np.diag( rhs )
        # Reshape into 2-matrix
        hess = np.reshape( hess, (self.n_variables,self.n_variables) )
        return hess[ self.hessianstructure() ]
            
   
    
    def solve(self):
        # Set the starting point
        st = self.getStartingPoint() 
        x, info = super().solve(st)
        print (info['status_msg'])
        # Copy to the results dictionary
        self.results = dict()           # Reset
        keys = ['status','x','u','obj','lagrange','summary', 'grid', 'start']
        entries = [ info['status_msg'], x, self.getU(x), info['obj_val'], info['mult_g'], str(self), self.grid, st  ]
        for k, ee in zip(keys, entries):
            self.results[k] = ee
        # Save the dictionary to file
        saveData(self.datafile, self.results)
        # Print to file
        with open(self.f_file, 'w') as fx:
            with open(self.u_file, 'w') as fu:
                u = self.getU(x)
                x = np.reshape(x, (self.n_orbitals,self.n_points) )
                for q in range(self.n_orbitals):
                    for ff in (fx,fu):
                        ff.write("# {n}\n".format(n=self.orbital_set[q].getName()))
                    for rr, xx, uu in zip(self.grid, x[q,:], u[q,:] ):
                        fx.write("{rr:.2f}\t{xx:.10E}\n".format(rr=rr,xx=xx))
                        fu.write("{rr:.2f}\t{uu:.10E}\n".format(rr=rr,uu=uu))
        # Return results dictionary and ipopt info message
        return self.results, info
    
    
    """
    Intermediate callback
    """
    def intermediate( self, alg_mod, iter_count, obj_value,   \
            inf_pr, inf_du, mu, d_norm, regularization_size,  \
            alpha_du, alpha_pr, ls_trials):
        if iter_count%10==0:
            print ("Objective value at iteration #%d is %g" % (iter_count, obj_value))
                   
                   
                   
      
    def _setUp(self,max_iter,rel_tol,constr_viol):      
        # Options
        self.max_iter = max_iter
        self.rel_tol = rel_tol
        self.constr_viol = constr_viol
        # Calling ipopt.problem constructor
        ub_x = np.ones(self.n_variables)
        lb_x = -1. * ub_x
        con = self._getConstraintsValue()
        super().__init__(n=self.n_variables, m=self.n_constr, lb=lb_x, ub=ub_x, cl=con, cu=con)
        # Set options
        self.setSolverOptions()
        
        
        
    """
    Returns a summary of the problem 
    """
    def __str__(self):
        st = "Z={Z}\tN={N}\n".format(Z=self.Z, N=self.N)
        st += "Interval [{lb}:{ub}]\th={h}\n".format(lb=self.lb,ub=self.ub,h=self.h)
        if len(self.output_folder)>0:
            st += "Output directory: {dd}\n".format(dd=self.output_folder) 
        st += "N. points={np}\tN. constraints={nc}\n\nN.orbitals={nv}\n".format(np=self.n_points,nc=self.n_constr,nv=self.n_orbitals)
        st += str(self.orbital_set) +"\n"
        return st
            
    
    
    """
    Returns the list of orbital pairs (i,j) whose overlap <i|j> must be constrained
    """
    def getPairs(self):
        ll = []
        # If just one orbital, no need to impose <i|i>=1
        if self.n_orbitals==1:
            return []
        for i in range(self.n_orbitals):
            o_i = self.orbital_set[i]
            for j in range(i+1):
                o_j = self.orbital_set[j]
                # Check same l and same j (otherwise, <i|j>=0 automatically )
                if o_i.l==o_j.l and o_i.j==o_j.j:
                    ll.append( (i,j) )
        return ll
    
    
    def getRhoFromData(self, r, rho):
        ff, d1, d2 = getRhoFromData(r, rho)
        return ff
    
    
    """
    Return C0, C1, C2
    """
    def _getCFunctions(self):
        C0 = np.zeros( (self.n_orbitals,self.n_points) )
        # cc = self.grid*self.d1_rho + self.grid**2/2. *self.d2_rho - self.grid**2/(4.*self.tab_rho) * self.d1_rho**2
        cc = self.grid**2/2. * self.d2_rho - np.power(self.grid*self.d1_rho, 2)/(4.*self.tab_rho) + self.grid * self.d1_rho
        for j in range(self.n_orbitals):
            l = self.orbital_set[j].l
            C0[j, :] = cc - l*(l+1)* self.tab_rho
        # rho(r) 2r + r^2 drho/dr;     C1=d/dr(C2)
        C1 = 2.*self.grid*self.tab_rho  +  self.grid**2 * self.d1_rho
        # rho(r) r^2
        C2 = self.grid**2 * self.tab_rho
        return C0, C1, C2
    
    
    def _getConstraintsValue(self):
        arr = np.ones(self.n_constr)
        for k, (i,j) in enumerate(self.pairs):
            # Enforce <i|j>=0  (if i!=j)
            if i != j:
                arr[self.n_points + k] = 0.
        return arr
    
    """
    Start from HO orbitals
    """
    def getStartingPoint(self):
        st = np.zeros( shape=(self.n_orbitals, self.n_points) )
        self.orbital_set.reset()
        for j, oo in enumerate(self.orbital_set):
            #print (oo.name)
            # print (j, str(oo))
            wf = HO_3D( oo.n, oo.l, self.nu)   # R(r) (=u(r)/r)
            # f(r) = u(r)/r / sqrt(4 pi rho(r) )
            st[j,:] = wf(self.grid)/np.sqrt( 4.*np.pi*self.tab_rho )
        return np.ndarray.flatten(st)
    
    
    def setSolverOptions(self):
        """
        Solver options: relative tolerance on the objective function;
        absolute tolerance on the violation of constraints;
        maximum number of iterations
        """
        # Watch out! Put b (binary) in front of option strings    
        self.addOption(b'mu_strategy', b'adaptive')
        self.addOption(b'max_iter', self.max_iter)
        self.addOption(b'tol', self.rel_tol)
        self.addOption(b'constr_viol_tol', self.constr_viol)
        self.addOption(b"output_file", b"ipopt.out")
        #self.addOption(b"hessian_approximation", b"exact")
        self.addOption(b'hessian_approximation', b'limited-memory')
        
        
        
    """
    Determine the reduced wave functions u(r) from the x array.
    Returns the matrix [u_1, .. ,u_n]
    """
    def getU(self, x):
        # u(r) = sqrt(4 pi rho(r) ) * r * f(r)
        x = np.reshape(x, (self.n_orbitals, self.n_points) )
        u = np.zeros_like(x)
        for j in range(self.n_orbitals):
            u[j,:] = x[j,:] * self.grid * np.sqrt( 4.*np.pi * self.tab_rho )
        return u    
    
    
    def integrate(self, f):
        # assert ( f.shape[0]==self.h_i.shape[0] )
        return np.sum( f*self.h_i )
    
    
    """
    Return [f'_1, ..., f'_n] and [f''_1, ..., f''_n]
    """
    def _deriv(self,x):
        d1x = np.zeros_like(x)
        d2x = np.zeros_like(x)
        for j in range(self.n_orbitals):
            d1x[j,:] = self.d_dx(  x[j,:] )
            d2x[j,:] = self.d_d2x( x[j,:] )
        return d1x, d2x
    
    
    
    
        
    
    
    
    
    
    
    
    
"""
Interpolates (r, rho(r) ) arrays and returns the interpolating spline and its derivatives
"""
def getRhoFromData(r, rho):
    from scipy import interpolate
    assert( len(r)==len(rho) )
    r=np.array(r); rho=np.array(rho)
    tck  = interpolate.splrep(r, rho)
    ff = lambda x: interpolate.splev(x, tck )
    d1 = lambda x: interpolate.splev(x, tck, der=1 )
    d2 = lambda x: interpolate.splev(x, tck, der=2 )
    return ff, d1, d2



def quickLoad(file="Densities/SkXDensityCa40p.dat"):
        # file = "Densities/SOGDensityPb208p.dat"
        file = open(file)
        ff = file.readlines()
        file.close()
        r = []; dp = []
        for ll in ff:
            if str(ll).startswith("#"):
                  pass
            else:
                ll = [ float(x) for x in ll.split() ]
                r.append( ll[0] )
                dp.append( ll[1] )
        r=np.array(r); dp=np.array(dp)
        return (r,dp)
    
    
"""
Returns the density function rho(r) of a set of n_orb harmonic oscillators 
"""
def getSampleDensity(n_orb, basis=ShellModelBasis() ):
    basis = OrbitalSet([ c for c in basis[:n_orb] ])
    #print ("Basis\t", basis)
    n_part = basis.countParticles()
    nu = nuclearNu(n_part)
    wf = []
    for j, oo in enumerate(basis):
        wf.append( HO_3D(oo.n, oo.l, nu) )
    # wf(r) = R(r) = u(r)/r
    def rho(r):
        arr = np.array( [basis[j].occupation * wf[j](r)**2 for j in range(n_orb)] )
        return np.sum(arr,axis=0)/(4.*np.pi)
    return rho
        

"""
Things to do or check:
    - integral_j: two formulas which should be equivalent, but are not
    - gradient: *h or not? (Probably yes)
    - jacobian: check if in the deriv. of the ortho. must put *h
    - Hessian:  check everything
    - save output in a dictionary or file
    - understand coupled vs. uncoupled basis
    - compute the potential
    
    - is the spline a good way to interpolate the density?
    - check "solve" matrix
"""

if __name__=="__main__":
    
    dummy = Problem(20,20,data=quickLoad() )
    # print (dummy,"\n\n")
    # gr = nucl.gradient(x0)
    x0 = dummy.getStartingPoint()
    u0 = dummy.getU(x0)
    
    
    nucl = Problem(20,20,data=quickLoad(),max_iter=4000, debug='y', basis=ShellModelBasis(), rel_tol=1e-4, constr_viol=1e-4  )
    # nucl = Problem(Z=20,N=20,max_iter=4000, ub=8., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat"), constr_viol=1e-4 )
    print (nucl)
    data, info = nucl.solve()
    data = loadData(nucl.output_folder+"\data")
    x = data['x']
    
   
   