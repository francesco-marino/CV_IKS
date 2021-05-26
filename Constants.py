# -*- coding: utf-8 -*-

" Useful physical constants "

from numpy import pi, power, array, sum, linspace, zeros_like

# Physical constants
hbar = 197.327053          # MeV*fm  (hc)
m    = 938.91869           # MeV     (mc^2)
c    = 3.*10**(23)         # fm/s

e = 1.2
r0 = 1.25

coeffSch = hbar**2/(2*m)
T = hbar**2/(2.*m) *(4.*pi)

# Nuclear HO frequency
def nuclearOmega(n):
    return  41./( hbar * power(n, 1./3) )

def nuclearNu(n):
    return nuclearOmega(n)*m/(2.*hbar)



        
        
    
