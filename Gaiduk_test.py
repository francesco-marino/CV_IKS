# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 20:32:01 2021

@author: alberto
"""
import numpy as np
from Energy import Energy


"""
Parametric potentials from input
"""
    
def param_Potential(self):
    r = self.potential[0]
    p = self.potential[1]
        
    v=[]
    for t in self.T:
        v.append(t**2*p)
        
    t_col=np.reshape(self.T,newshape=(-1,1))
    print("\n t_col \t", np.shape(t_col))
    
    return v, v/(3*t_col), v/(2*np.sqrt(t_col)), r
    
#%%
import numpy as np
from Energy import Energy

r = np.array([r for r in np.arange(0,15,0.1)])
rho = np.ones(shape=r.shape)
potential = rho

#print(r, rho, potential)

energy = Energy(Z=2, potential=(r,potential), data=(r,rho), param_step=0.1, r_step=0.001)
E = energy.solver()
print(E)

#OK: 
#param_step = 0.001: 2510 - 4188 - 3131
#param_step = 0.1: 2303 - 4188 - 2975
#%%
import numpy as np
from Energy import Energy

r = np.array([r for r in np.arange(0,15,0.1)])
rho = r
potential = rho**2

#print(r, rho, potential)

energy = Energy(Z=2, potential=(r,potential), data=(r,rho), param_step=0.01, r_step=0.01)
E = energy.solver()
print(E)

#OK 14226 - 19999 - 16560


#%%
#####################################OTHER
basis = ShellModelBasis()
        print(ShellModelBasis())
        print(basis)
        print("??????",basis==ShellModelBasis())
        print("!!!!!!!!!!!!!!!!!!",self.basis == basis)