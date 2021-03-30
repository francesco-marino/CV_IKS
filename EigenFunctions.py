# -*- coding: utf-8 -*-


"""
Special functions; useful eigenfunctions
"""

import numpy as np
from math import factorial



def Laguerre(n,A,z):
    if n==0:
        return 1.
    elif n==1:
        return 1+A-z
    else:
        return ((2*n-1-z+A)*Laguerre(n-1,A,z) - (n+A-1)*Laguerre(n-2,A,z) )/n
    

def dfactorial(n):
    if n==0 or n==1:
        return 1
    else:
        return n*dfactorial(n-2)
    
    
# harmonic oscillator 3D eigenfunctions
def HO_3D(N,L,nu):
    norm = np.sqrt( 2*np.power(nu,3)/np.pi )    \
        * np.power(2,N+2*L+2)                   \
        * factorial(N-1) * np.power(nu,L)       \
        / dfactorial(2*N+2*L-1) 
    def f(x):
        R = np.power(x,L) * np.exp(-nu*x*x) * \
            Laguerre(N-1,L+0.5, 2.*nu*np.power(x,2) )
        return (R*np.sqrt(norm))
    
    return f
        
    