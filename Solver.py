# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:26:50 2020

@author: Francesco
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, lsqr
import matplotlib.pyplot as plt

from Problem  import Problem, getSampleDensity, quickLoad
from Orbitals import UncoupledHOBasis, ShellModelBasis
from Constants import coeffSch
from Misc import loadData


class Solver(object):
    
    def __init__(self, problem, x=[] ):    
        assert ( isinstance(problem, Problem) )
        self.problem = problem
        self.data = loadData( problem.datafile )
        self._getInfoFromProblem()
        
        # Orbitals
        x = np.array(x)
        self.x = self.data['x'] if x.shape[0]==0 else x
        self._getOrbitals(self.x)
    
    
    def _getOrbitals(self, x):
        self.u, self.du, self.d2u = self.getU(x)
        self.b = self._getB()
        self.A = self._getA()
    
    
    def _getB(self):
        _len = self.n_points*self.n_orbitals*(self.n_orbitals+1)//2
        B = np.zeros( _len )    #(_len, self.n_points) )
        for a in range(self.n_orbitals):
            for b in range(a+1):
                if (a,b) in self.pairs:
                    l = self.orbital_set[a].l
                    row = (a*(a+1)//2 +b)*self.n_points
                    B[row:row+self.n_points] = coeffSch * self.u[b,:]* ( self.d2u[a,:] -l*(l+1)/self.grid**2 * self.u[a,:] )
        return np.ndarray.flatten(B)
    
    def _getA(self):
        _len = self.n_points*self.n_orbitals*(self.n_orbitals+1)//2
        _n_constr = self.n_points + self.n_orbitals*(self.n_orbitals+1)//2
        A = np.zeros( (_len,_n_constr) )
        for a in range(self.n_orbitals):
            for b in range(a+1):
                if (a,b) in self.pairs:
                    row = (a*(a+1)//2 +b)*self.n_points
                    # fill (row+i; i)
                    A[row:row+self.n_points, :self.n_points] = np.diag( self.u[b,:]*self.u[a,:]*2. )
        # Epsilon
        for a in range(self.n_orbitals):
            for b in range(a+1):
                if (a,b) in self.pairs:
                    for p in range(self.n_orbitals):
                        for q in range(p+1):
                            if a==p or a==q:
                                for i in range(self.n_points):
                                    row = (a*(a+1)//2 +b)*self.n_points 
                                    col = self.n_points + (p+1)*p//2 + q
                                    if q<a and self.orbital_set[q].l==self.orbital_set[b].l and self.orbital_set[q].j==self.orbital_set[b].j:
                                        A[row:row+self.n_points, col] = - self.u[b,:]* self.u[q,:]
                                    if p>a and self.orbital_set[p].l==self.orbital_set[b].l and self.orbital_set[p].j==self.orbital_set[b].j:
                                        A[row:row+self.n_points, col] = - self.u[b,:]* self.u[p,:]
                                    if p==q and self.orbital_set[p].l==self.orbital_set[b].l and self.orbital_set[p].j==self.orbital_set[b].j:
                                        A[row:row+self.n_points, col] = - self.u[b,:]* self.u[p,:]
        return A
    
    
   
    """
    Get u, du/dr, d2u/dr2 (in matrix form) from an array x
    """
    def getU(self, x):
        x = np.array(x)
        u = self.problem.getU(x)
        du = np.zeros_like(u)
        d2u= np.zeros_like(u)
        for j in range(self.n_orbitals):
            du[ j,:]= self.d_dx( u[j,:])
            d2u[j,:]= self.d_d2x(u[j,:])
        return u, du, d2u
    
    def _getVandE(self, lambd):
        return lambd[:self.n_points], lambd[self.n_points:]
    
    """
    Copy some important variables from problem 
    """
    def _getInfoFromProblem(self):
        self.n_orbitals = self.problem.n_orbitals
        self.orbital_set = self.problem.orbital_set
        self.n_points = self.problem.n_points
        self.n_constr = self.problem.n_constr
        self.pairs= self.problem.pairs
        self.grid = self.problem.grid
        # Derivative operators
        self.d_dx = self.problem.d_dx
        self.d_d2x = self.problem.d_d2x
        # Output files
        self.pot_file = self.problem.pot_file
        self.epsilon_file  = self.problem.epsilon_file
  
  
    
    
    def solve(self):
        print ("A\t",self.A.shape,"\tb\t",self.b.shape)
        A_sp = csc_matrix(self.A, dtype=float)
        # x = spsolve(A_sp, self.b)
        x, istop, itn, r1norm = lsqr(A_sp, self.b, atol=1e-9, btol=1e-9)[:4]
        check = np.allclose( A_sp.dot(x), self.b )
        # Write to file
        with open(self.pot_file, 'w') as fv:
            v,eps = self._getVandE(x)
            for rr, vv in zip(self.grid, v):
                fv.write("{rr:.2f}\t{vv:.10E}\n".format(rr=rr, vv=vv) )
        with open(self.epsilon_file, 'w') as fe:
            v,eps = self._getVandE(x)
            for k, (i,j) in enumerate(self.pairs):
                fe.write("{ni}\t{nj}\t{ep:.10E}\n".format(ni=self.orbital_set[i].getName(), nj=self.orbital_set[j].getName(), ep=v[k]))
        return x, check
    
    
    
    def getPotential(self):
        x, check = self.solve()
        v,eps = self._getVandE(x)
        return v
        
        
        



if __name__=="__main__":
    
    basis = UncoupledHOBasis()
    n_orb = 6
    rho = getSampleDensity(n_orb, basis=basis )
    
    #nucl = Problem( OrbitalSet(basis[:n_orb]).countParticles(), rho=rho,basis=basis)
    nucl = Problem(Z=20,N=20,max_iter=4000, ub=10., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityCa40p.dat") )
    #nucl = Problem(Z=20,N=20,max_iter=4000, ub=6., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat"), constr_viol=1e-4 )
    data, info = nucl.solve()
    data = loadData(nucl.datafile)
    x0 = nucl.getStartingPoint()
    
    x = data['x']
    s0 = Solver(nucl, x0)
    solver = Solver(nucl, x)
    x, check = solver.solve()
    print (check)
    
    plt.figure(0)
    plt.plot(solver.grid, solver.getPotential(), '--', label="Solution")
    plt.plot(s0.grid, s0.getPotential(), ls='-.', c='r',  label="Init")
    plt.grid(); plt.legend()
    
   
  