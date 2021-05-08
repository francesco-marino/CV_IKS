# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:36:28 2021

@author: alberto
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
    
def PlotInColumns(x1,y1,x2,y2,title):
    
    fig = plt.figure(figsize=(10,8))
    
    #set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    sns.set_theme(style='white',font_scale=3)#, palette = 'Pastel2')

    #first plot
    ax1 = plt.subplot(gs[0])
    ax1.plot(x1, y1,color='purple', linewidth=3)
    
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    #second plot
    ax2 = plt.subplot(gs[1], sharex = ax1)
    ax2.plot(x2, y2,color='darkorange', linewidth=3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    #labels
    ax2.set_xlabel("r")
    ax1.set_ylabel(r'$\rho$(r)')
    ax2.set_ylabel(r"v([$\rho$],r)")
    ax1.set_title(title)
    #grid
    ax1.grid()
    ax2.grid()
    
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    
    fig.savefig("test.png", bbox_inches='tight')

r=np.arange(0.,10.,0.1)
d=np.ones_like(r)
p=np.ones_like(r)
PlotInColumns(r, d, r, p)


##How to hide some labels
"""
#first method (not working if #ticks<3)
ax1.set_yticks(ax1.get_yticks())
labels = ax1.get_yticklabels()
# remove the first and the last labels
labels[0] = ""
ax1.set_yticklabels(labels)
OR
# plt.setp(ax1.get_yticklabels()[0], visible=False) 

#alternative:
    
ticks_loc = ax1.get_yticks().tolist()
ticks_loc.pop(0)
ticks_loc.pop(-1)
ax1.set_yticks(ticks_loc)
"""