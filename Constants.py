# -*- coding: utf-8 -*-

" Useful physical constants "

from numpy import pi, power, array, sum, linspace, zeros_like

# Physical constants
hbar = 197.3269788           # MeV*fm  (hc)
m_p  = 939.5654133           # MeV     (mc^2)
c    = 3.*10**(23)           # fm/s

e = 1.2
r0 = 1.25

coeffSch = hbar**2/(2*m_p)
T = hbar**2/(2.*m_p) *(4.*pi)

# Nuclear HO frequency
def nuclearOmega(n):
    return  41./( hbar * power(n, 1./3) )

def nuclearNu(n):
    return nuclearOmega(n)*m_p/(2.*hbar)



        
        
    
