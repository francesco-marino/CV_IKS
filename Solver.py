# -*- coding: utf-8 -*-



import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, lsqr
from scipy import integrate
import matplotlib.pyplot as plt

from Problem  import Problem, quickLoad
from Orbitals import UncoupledHOBasis, ShellModelBasis, OrbitalSet
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
        
        # Square matrix of epsilon multipliers
        self.eps_matrix = np.zeros( shape=(self.n_orbitals,self.n_orbitals) )
        # Sorted eigenvalues and orbitals
        #self.sorted_orbital_set = None
        self.eigenvalues = None
        # Matrix of transformed orbitals (same shape as u)
        self.eigenvectors= None
    
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
                B[r]   = self.coeffSch * ( self.d2u[k,p] -l*(l+1)/self.grid[p]**2 * self.u[k,p] )
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
                            A[r,self.n_points+a] += -self.u[not_k,p]
                # Update row number 
                r += 1
        return A, B
        
   
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
    """
    def updateU(self, new_u):
        assert (new_u.shape==self.u.shape)
        u, du, d2u = np.zeros_like(self.u), np.zeros_like(self.u), np.zeros_like(self.u)
        for j in range(self.n_orbitals):
            u[j,:] = new_u[j,:]
            du[ j,:]= self.d_dx( u[j,:])
            d2u[j,:]= self.d_d2x(u[j,:])
        self.u, self.du, self.d2u = u, du, d2u
        self._AB()
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
        self.potential = lambd[:self.n_points]
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
        self.coeffSch = self.problem.coeffSch
        self.kinetic = self.problem.kinetic
        self.kinetic_density = self.problem.kinetic_density
        
        # Derivative operators
        self.d_dx = self.problem.d_dx
        self.d_d2x = self.problem.d_d2x
        # Output files
        self.pot_file = self.problem.pot_file
        self.epsilon_file  = self.problem.epsilon_file
        self.eigen_file = self.problem.eigen_file
  
  
    
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
    diag: bool
        compute the energy eigenvalues and the eigenvectors (rotated orbitals) (default True)
    shift: bool
        shift the potential and the eigenvalues by a costant (default True)
    cost: float
        shift pot. and eps. by -cost. If None, a simple ansatz is used
        
    Returns
    ----------
    x: np.array(n_constr)
        Lagrange multipliers (v(r),epsilon)
    
    """
    def solve(self, max_prec=True, tol=1e-10, max_iter=10000, diag=True, shift=True, cost=None):
        # Build sparse matrix
        self.A_sparse = csc_matrix(self.A, dtype=float)
        # Solve with least-squares
        if max_prec:
            x, istop, itn, r1norm = lsqr(self.A_sparse, self.b, atol=0., btol=0., conlim=0., show=True, iter_lim=50000)[:4]
        else:
            x, istop, itn, r1norm = lsqr(self.A_sparse, self.b, atol=tol, btol=tol, iter_lim=max_iter)[:4]
        #check = np.allclose( self.A_sparse.dot(x), self.b )
        
        v, eps = self._getVandE(x)
        for k, (i,j) in enumerate(self.pairs):
            # Fill epsilon matrix (symmetric)
            self.eps_matrix[i,j] = eps[k]
            self.eps_matrix[j,i] = eps[k]
                
        if diag:
            self.diagonalize(eps=self.eps_matrix)
        if shift: 
            self.shiftPot(cost)
        
        # Compute int(rho*v)
        self.int_rho_v = integrate.simps(self.problem.tab_rho*self.potential*self.grid**2, self.grid) * 4.*np.pi
 
        
        # Write potential to file
        with open(self.pot_file, 'w') as fv:   
            for rr, vv in zip(self.grid, self.potential):
                fv.write("{rr:.2f}\t{vv:.10E}\n".format(rr=rr, vv=vv) )
        # Write epsilon matrix
        with open(self.epsilon_file, 'w') as fe:
            for k, (i,j) in enumerate(self.pairs):
                fe.write("{ni}\t{nj}\t{ep:.10E}\n".format(ni=self.orbital_set[i].getName(), nj=self.orbital_set[j].getName(), ep=eps[k]))
        # Write eigenvalues
        with open(self.eigen_file, 'w') as fe:
            for j in range(self.n_orbitals):
                fe.write("{nj}\t{ep:.10E}\n".format(nj=self.orbital_set[j].getName(), ep=self.eigenvalues[j]))
                #print ( str(self.orbital_set[j]), "\t", self.eigenvalues[j] )
        return x
    
    
    """
    Yields the potential
    
    Returns
    ----------
    v: np.array(n_points)
        the potential
    """
    def getPotential(self, shift=True, cost=None):
        x, check = self.solve()
        v,eps = self._getVandE(x)
        if shift: self.shiftPot(cost)
        return v
    
        
    
    """
    Diagonalizes the matrix eps of multipliers.
    Returns the energy eigenvalues and a set of tranformed orbitals,
    which are eigenvectors of the energy.
    Moreover, the resulting orbitals are sorted according to their energy, as
    in standard Hartree-Fock or Schroedinger eqs.
    """
    def diagonalize(self, eps=None):
        if eps is None:
            eps = self.eps_matrix
        # Find subspaces and subsets of eps with same l and j (dictionaries)
        subspaces = dict()
        submatrices = dict()
        # key: (l,j); value: list of orbital indeces   
        self.problem.orbital_set.reset()
        for k, q in enumerate(self.problem.orbital_set):
            lj = (q.l, q.j)
            if not (lj in subspaces.keys() ):
                subspaces[ lj ] = []
            # k-th orbitals belongs to lj subspace
            subspaces[ lj ].append(k)
        # Find submatrix of epsilon corresponding to given (l,j)
        for lj, val in subspaces.items():
            submatr = []
            for row in val:
                for col in val:
                    submatr.append( eps[row,col] )
            submatr = np.reshape(submatr, (len(val),len(val)) )
            submatrices[ lj ] = submatr
        # Diagonalize each submatrix 
        orb_num, eigenvalues = [], []
        new_u = self.u.copy()
        for lj, matr in submatrices.items():
            if matr.shape[0] > 1:
                vals, vect = np.linalg.eig( matr )
                # sort eigenvalues and eigenvectors 
                sorted_indexes = np.argsort(vals)
                vals = vals[sorted_indexes]
                vect = vect[:,sorted_indexes]
                # Apply orthogonal tranformation R to orbitals
                # matr = vect @ vals @ inv(vect) => u's traform with inv(vect)
                R = np.linalg.inv( vect )   # rotation matrix
                for mi, m in enumerate(subspaces[ lj ]):
                    new_u[m,:] = 0.
                    for ni, n in enumerate(subspaces[ lj ]):
                        new_u[m,:] += R[mi,ni]* self.u[n,:]
            else:
                vals = [matr[0,0],]
            # Save eigenvalues and corresponding orbital number
            for k, v in enumerate(vals):
                eigenvalues.append(v)
                orb_num.append( subspaces[lj][k] )
                
        # Sort ALL orbitals and eigenvalues
        sorted_indexes = np.argsort(eigenvalues)
        self.eigenvalues = np.array(eigenvalues)[sorted_indexes]
        self.sum_eigenvalues = np.sum([self.orbital_set[k].occupation*self.eigenvalues[k]\
            for k in range(self.eigenvalues.shape[0])])
        orb_num = np.array(orb_num, dtype=int)[sorted_indexes]
        # Subscribe orbitals set
        self.orbital_set.reset()
        new_set = OrbitalSet([self.orbital_set[ oo ] for oo in orb_num])
        self.orbital_set = new_set
        #self.sorted_orbital_set = OrbitalSet([self.orbital_set[ oo ] for oo in orb_num])
        # Subscribe u and its derivatives
        self.eigenvectors = new_u[orb_num, : ]
        u, du, d2u = self.updateU(self.eigenvectors)      
        return self.eigenvalues, self.eigenvectors
    
    """
    Shift the potential and the eigenvalues by  -cost
    """
    def shiftPot(self, cost=None):
        if cost is None:
            cost = self.potential[-10]
        print ("shift   ", cost )
        self.potential -= cost
        self.eigenvalues -= cost
        self.sum_eigenvalues = np.sum([self.orbital_set[k].occupation*self.eigenvalues[k]\
            for k in range(self.eigenvalues.shape[0])])
    
    
    def printAll(self):
        out = self.problem.output_folder + "/Energies.dat"
        with open(out, 'w') as fu:
            st = "Kin\tsum(eps)\tint(rho*v)\n"
            fu.write(st)
            int_rhov = integrate.simpson(self.problem.tab_rho*self.potential*self.grid**2, self.grid) * 4.*np.pi
            st = "{t:.5f}\t{eps:.5f}\t{integ:.5f}\n".format(t=self.problem.kinetic, eps=self.sum_eigenvalues, integ=int_rhov)
            fu.write(st)
        return self.problem.kinetic, self.sum_eigenvalues, int_rhov
        
        
  
        
        
        
        
        
        
"""   
def kin_energy(solver):
    assert ( isinstance(solver, Solver) )
    tau = np.zeros_like(solver.grid)
    kin = 0.
    cp = 0.079577471
    UNS4PI =7.9577470999999997/100.   # 1/(4pi)
    print (UNS4PI)

    for j in range(solver.n_points):
        te = 0.
        for i in range(solver.n_orbitals):
            unl = solver.u[i,j]
            x = solver.grid[j]
            ll1 = solver.orbital_set[i].l*(solver.orbital_set[i].l +1 )
            dunl = solver.du[i,j]
            y = (-unl/x + dunl)/x
            #y = dunl/x
            y1= unl/x**2
            y2 = (y*y + ll1*y1*y1) * solver.orbital_set[i].occupation
            te += y2  
        # loop sui punti
        tau[j] = te * UNS4PI
        kin += tau[j] * x**2 
    # integral
    hb = 19.438109125130669
    
    kin = kin*4.*np.pi * solver.problem.h *hb 
    #kin = np.sum(tau * solver.grid**2) * 4.*np.pi * solver.problem.h * hb 
    return tau, kin
"""          
            



if __name__=="__main__":
    #nucl = Problem(Z=20,N=20, n_type='p', max_iter=4000, ub=11.4, debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/rho_ca40_t0t3.dat"), exact_hess=True )
    #nucl = Problem(Z=20,n_type='p', max_iter=4000, ub=8., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/rho_HO_20_particles_coupled_basis.dat") )
    nucl = Problem(Z=8,N=8, n_type='p', h=0.1, max_iter=4000, ub=11, debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/rho_o16_t0t3.dat"), exact_hess=True )
    #nucl = Problem(Z=82,N=106, n_type='p', max_iter=4000, ub=12., debug='y', basis=ShellModelBasis(), data=quickLoad("Densities/SOGDensityPb208p.dat"), exact_hess=True )
   
    results, info = nucl.solve()
    
    solver = Solver(nucl)
    
    # Benchmark
    out = read("Potentials/pot_ca40_t0t3.dat")
    r, vp = out[0], out[1]
    
    plt.figure(0)
    pot = solver.getPotential()
    plt.plot(solver.grid, pot , '--', label="CV")
    #plt.plot(solver.grid, pot - pot[3]+vp[3], '--', label="CV")
    #plt.plot(r, vp, label="exact")
    plt.xlim(0.,10.); plt.ylim(-70.,25.)
    plt.grid(); plt.legend()
    
    
    u = solver.u
    plt.figure(1)
    for j in range(u.shape[0]):
        #plt.plot(nucl.grid, u[j,:], ls='--', label=nucl.orbital_set[j].name)
        plt.plot(solver.grid, solver.eigenvectors[j,:], ls='--', label=solver.orbital_set[j].name)
    plt.legend()
    
    
    solver.printAll()
    
  
    """
    tau, kin = kin_energy(solver)
    print ("Kinetic energy   ", kin)
    plt.figure(20)
    plt.plot(solver.grid, tau, '--', label="Tau")
  
    
    from Misc import read
    r, tau_t0t3, tau_r2 = read("tau_p_o16_t0t3.out")
    plt.plot(r, tau_t0t3, label="HF")
    plt.xlim(0.,5.)
    plt.legend(); plt.grid()
    """
    
    
    
   
  
