# -*- coding: utf-8 -*-



import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, lsqr
import matplotlib.pyplot as plt

from Problem  import Problem, quickLoad
from Orbitals import UncoupledHOBasis, ShellModelBasis
from Constants import coeffSch
from Misc import loadData, read


class Solver(object):
    
    """
    A class to determine the potential and eigenvalues of an IKS problem,
    given a set of rescaled orbitals 

    ...

    Parameters
    ----------
    problem: Problem
        an instance of the class Problem
    
    x: np.array
        value of the orbitals (default: [])
        By default, 
        
    
    
    """
    
    def __init__(self, problem, x=[] ):    
        assert ( isinstance(problem, Problem) )
        self.problem = problem
        self.data = loadData( problem.datafile )
        self._getInfoFromProblem()
        
        # Orbitals
        x = np.array(x)
        self.x = self.data['x'] if x.shape[0]==0 else x
        self._getOrbitals(self.x)
    
    """
    """
    def _getOrbitals(self, x):
        self.u, self.du, self.d2u = self.getU(x)
        #self.b = self._getB(); self.A = self._getA()    # Old
        self.A, self.b = self._AB()                     # New
        
        
    """
    Define the matrix A and vector B.
    The Lagrange multipliers (potential + eigenvalues) are determined by solving
    the linear problem Ax=b (see solve).
    As A is not a squared matrix, the approximate solution is found with a least-square
    method.
    
    Returns
    ----------
    A: np.array(n_points*n_orbitals, n_constr)
    B: np.array(n_points*n_orbitals)
    
    """
    def _AB(self):
        _len = self.n_points*self.problem.n_orbitals
        # A -> row=(orbital,point); col=constr
        # B -> row=(orbital,point)
        # Ax=B -> x: row=constr 
        A = np.zeros( (_len, self.problem.n_constr) )
        B = np.zeros( _len )
        # Row index
        r = 0
        # k: orbital; p: point
        for k in range(self.problem.n_orbitals):
            l = self.orbital_set[k].l
            for p in range(self.n_points):
                # Rhs  (k,p)
                B[r]   = coeffSch * ( self.d2u[k,p] -l*(l+1)/self.grid[p]**2 * self.u[k,p] )
                # Density constraints (first n_points constraints)
                A[r,p] = self.u[k,p]
                # Orthonormality constr. 
                for a, (i,j) in enumerate(self.problem.pairs):
                    # Find orbitals "paired" with k
                    if k in (i,j):
                        not_k = j if k==i else i
                        if i==j:    # Normality
                            A[r,self.n_points+a] += -self.u[k,p]
                        else:       # Orthogonality     
                            A[r,self.n_points+a] += -0.5* self.u[not_k,p]

                    """
                    if k==i:
                        A[r,self.n_points+a] += - self.u[j,p]
                    """
                # Update row number 
                r += 1
        return A, B
        
        
                
                
            
            
                
        
        
    
    def _getB(self):
        _len = self.n_points*self.n_orbitals*(self.n_orbitals+1)//2
        B = np.zeros( _len )    #(_len, self.n_points) )
        for a in range(self.n_orbitals):
            for b in range(a+1):
                if (a,b) in self.pairs:
                    l = self.orbital_set[a].l
                    #row = (a*(a+1)//2 +b)*self.n_points
                    for i in range(self.n_points):
                        pos = self.n_points*(b + a*(a+1)//2 ) + i
                        B[pos] = coeffSch * self.u[b,i]* ( self.d2u[a,i] -l*(l+1)/self.grid[i]**2 * self.u[a,i] )
                    #B[row:row+self.n_points] = coeffSch * self.u[b,:]* ( self.d2u[a,:] -l*(l+1)/self.grid**2 * self.u[a,:] )
        return np.ndarray.flatten(B)
    
    def _getA(self):
        _len = self.n_points*self.n_orbitals*(self.n_orbitals+1)//2
        _n_constr = self.n_points + self.n_orbitals*(self.n_orbitals+1)//2
        A = np.zeros( (_len,_n_constr) )
        for a in range(self.n_orbitals):
            for b in range(a+1):
                if (a,b) in self.pairs:
                    for i in range(self.n_points):
                        row = (a*(a+1)//2 +b)*self.n_points + i
                        A[row,i] = self.u[b,i]*self.u[a,i]*2.
                    #row = (a*(a+1)//2 +b)*self.n_points
                    #A[row:row+self.n_points, :self.n_points] = np.diag( self.u[b,:]*self.u[a,:]*2. )
        # Epsilon
        for a in range(self.n_orbitals):
            for b in range(a+1):
                if (a,b) in self.pairs:
                    for p in range(self.n_orbitals):
                        for q in range(p+1):
                            if a==p or a==q:
                                for i in range(self.n_points):
                                    row = (a*(a+1)//2 +b)*self.n_points + i
                                    col = self.n_points + (p+1)*p//2 + q
                                    if q<a and self.orbital_set[q].l==self.orbital_set[b].l and self.orbital_set[q].j==self.orbital_set[b].j:
                                        A[row, col] = - self.u[b,i]* self.u[q,i]
                                    if p>a and self.orbital_set[p].l==self.orbital_set[b].l and self.orbital_set[p].j==self.orbital_set[b].j:
                                        A[row, col] = - self.u[b,i]* self.u[p,i]
                                    if p==q and self.orbital_set[p].l==self.orbital_set[b].l and self.orbital_set[p].j==self.orbital_set[b].j:
                                        A[row, col] = - self.u[b,i]* self.u[p,i]
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
    
    """
    Utility function.
    Split array of Lagrange multipliers into potential
    and energy eigenvalues
    
    Parameters
    ----------
    x: np.array(n_constr)
        array of Lagrange multipliers (v + epsilon)
    
    Returns
    ----------
    v: np.array(n_points)
        potential
    eps: np.array(n_constr-n_points)
        energy eigenvalues epsilon
    """
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
  
  
    
    """
    Solve a matrix problem for potential and eigenvalues
    
    Parameters
    ----------
    max_prec: bool
        solver the system at max possible precision (default True) (may take several seconds)
    tol: float
        precision parameter of the least-square solver (default 1e-5) (used only if max_prec is False)
    max_iter: int
        max. number of iterations (default 10000) (used only if max_prec is False)
    
    Returns
    ----------
    x: np.array(n_constr)
        Lagrange multipliers (v(r),epsilon)
    check: bool
        closure test of the numerical procedure
    """
    def solve(self, max_prec=True, tol=1e-10, max_iter=10000):
        # print ("A\t",self.A.shape,"\tb\t",self.b.shape)
        # Build sparse matrix
        self.A_sparse = csc_matrix(self.A, dtype=float)
        # Solve with least-squares
        #x, istop, itn, r1norm = lsqr(self.A_sparse, self.b, atol=1e-10, btol=1e-10)[:4]
        if max_prec:
            x, istop, itn, r1norm = lsqr(self.A_sparse, self.b, atol=0., btol=0., conlim=0., show=True, iter_lim=50000)[:4]
        else:
            x, istop, itn, r1norm = lsqr(self.A_sparse, self.b, atol=tol, btol=tol, iter_lim=max_iter)[:4]
        #self.everything =  lsqr(self.A_sparse, self.b, atol=0., btol=0.)
        check = np.allclose( self.A_sparse.dot(x), self.b )
        # Write potential to file
        with open(self.pot_file, 'w') as fv:
            v,eps = self._getVandE(x)
            for rr, vv in zip(self.grid, v):
                fv.write("{rr:.2f}\t{vv:.10E}\n".format(rr=rr, vv=vv) )
        # Write epsilon eigenvalues
        with open(self.epsilon_file, 'w') as fe:
            v,eps = self._getVandE(x)
            for k, (i,j) in enumerate(self.pairs):
                fe.write("{ni}\t{nj}\t{ep:.10E}\n".format(ni=self.orbital_set[i].getName(), nj=self.orbital_set[j].getName(), ep=v[k]))
        return x, check
    
    
    """
    Yields the potential
    
    Returns
    ----------
    v: np.array(n_points)
        the potential
    """
    def getPotential(self):
        x, check = self.solve()
        v,eps = self._getVandE(x)
        return v
        
        
        



if __name__=="__main__":
    nucl = Problem(Z=20,N=20, n_type='p', max_iter=4000, ub=15., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityCa40p.dat"), exact_hess=True )
    #nucl = Problem(Z=20,n_type='p', max_iter=4000, ub=8., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat") )
    #nucl = Problem(Z=8,N=8, n_type='p', max_iter=4000, ub=12., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/SkXDensityO16p.dat"), exact_hess=True )
    nucl = Problem(Z=82,N=108, n_type='p', max_iter=4000, ub=12., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/SOGDensityPb208p.dat"), exact_hess=True )
    
    results, info = nucl.solve()
    
    #nucl.setDensity(data=quickLoad("Densities/SkXDensityCa40p.dat"))
    
    
    solver = Solver(nucl)
    x, check = solver.solve()
    print (check)
    
    
    # Benchmark
    out = read("Potentials\pot_ca40_skx.dat")
    r, vp = out[0], out[1]
    
    
    out = read("Potentials\pot_ca40_skx_other_iks.dat")
    r_other, vp_other = out[0], out[1]
    
    plt.figure(0)
    pot = solver.getPotential()
    plt.plot(solver.grid, pot - pot[3]+vp[3], '--', label="CV")
    plt.plot(r, vp, label="exact")
    plt.plot(r_other, vp_other, label="other")
    plt.xlim(0.,10.); plt.ylim(-70.,25.)
    plt.grid(); plt.legend()
    
    
    u = nucl.results['u']
    plt.figure(1)
    for j in range(u.shape[0]):
        plt.plot(nucl.grid, u[j,:], ls='--', label=nucl.orbital_set[j].name)
        #plt.plot(nucl.grid, nucl.getU(nucl.results['start'])[j,:], label=nucl.orbital_set[j].name+"  INIT")
    plt.legend()
    
   
    
    
   
  