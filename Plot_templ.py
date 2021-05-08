# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 18:01:19 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt

from Problem import quickLoad

name = "Scaled_Potentials/Ca40t0t3_coul/Ca40t0t3_01_-8/Potentials/pot_L=1.000000_C++.dat"
r, v = quickLoad(name)

def Plot(r,v, x="", y="", title="", lab=""):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(
        r, v,
        lw = 2,
        label = lab
        )
    
    plt.grid(); plt.legend()
    ax.set_title(title)
    ax.set_xlabel(x)
    # ax.set_xlim([0, 12])
    # ax.set_ylim([-50, -100])
    ax.set_ylabel(y)