import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from random import *

Full = np.zeros((401, 231), dtype = np.double)
for i in range (0, 400):
	E = np.loadtxt("46Hz.txt", usecols=(i,), skiprows= 639, unpack =True )
	Full[i] = E 

fig, ax = plt.subplots(figsize = (4.62,8.02))
plt.tick_params(bottom = False, left = False)
plt.setp(ax.spines.values(), linewidth=2)
#plt.figure(figsize = (4.62,8.02))
plt.ylim(0,401)
plt.xlim(0,231)     
       #plt.title('State after t = %s MCS at' %i)
       #plt.xlabel('T/k = %s, with Etrans/k = 10.0, Strain of 1.0 '%temps_array[i])
cm = "jet"
im = plt.pcolormesh(Full, cmap = 'jet')
plt.tight_layout()
#ax.set_aspect('1')
divider = make_axes_locatable(ax)
ax.set_yticklabels([])
ax.set_xticklabels([])
cax = divider.append_axes("bottom", size="5%", pad=0.5)
cbar = fig.colorbar(im, cax = cax, orientation='horizontal')
cbar.ax.set_ylabel('E Field Magnitude', orientation )      
plt.show()
