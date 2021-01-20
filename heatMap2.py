import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspe

from random import *

Full46Hz = np.zeros((401, 231), dtype = np.double)
#Full47Hz = np.zeros((401, 231), dtype = np.double)
X = np.linspace(0,  1.15, 231)
Y = np.linspace(0,  1.991858428704, 401)
for i in range (0, 400):
	E = np.loadtxt("46Hz.txt", usecols=(i,), skiprows= 639, unpack =True )
	Full46Hz[i] = E 

# for i in range (0, 400):
# 	E = np.loadtxt("47Hz.txt", usecols=(i,), skiprows= 639, unpack =True )
	#Full47Hz[i] = E 

#norm = mcolors.Normalize(vmin=0., vmax=100.)
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}#, 'norm': norm}
fig, ax0 = plt.subplots(figsize = (4, 4),constrained_layout=True)#figsize = (4.62,8.02))
ax0.set_aspect(401/231)

im = ax0.pcolormesh(X, Y, Full46Hz, **pc_kwargs)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.01)
fig.colorbar(im, ax=ax0, shrink = 0.6, label = r"$\rm E \ \ Field\ \ V m^{-1}$")
# cbar.ax0.set_ylabel('E Field Magnitude') 

ax0.set_title('46 Hz')
ax0.set_xlabel(r"$ \mu \rm m$")
ax0.set_ylabel(r"$ \mu \rm m$")
#ax0.set_yticklabels([])
#ax0.set_xticklabels([])
plt.setp(ax0.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)

#################################################################
# im1 = ax1.pcolormesh(Full47Hz, cmap = cm)
# fig.colorbar(im1, ax=ax1)
# ax1.set_title('47 Hz')

# fig.tight_layout()
plt.savefig("hBN_Ag_Si46Hz.pdf")
plt.savefig("hBN_Ag_Si46Hz.png")
plt.show()


# plt.tick_params(bottom = False, left = False)
# plt.setp(ax.spines.values(), linewidth=2)
# #plt.figure(figsize = (4.62,8.02))
# plt.ylim(0,401)
# plt.xlim(0,231)     
#        #plt.title('State after t = %s MCS at' %i)
#        #plt.xlabel('T/k = %s, with Etrans/k = 10.0, Strain of 1.0 '%temps_array[i])
# cm = "jet"
# im = plt.pcolormesh(Full, cmap = 'jet')
# plt.tight_layout()
# #ax.set_aspect('1')
# divider = make_axes_locatable(ax)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# width = axes_size.AxesY(ax, aspect=1./aspect)
# pad = axes_size.Fraction(pad_fraction, width)
# cax = divider.append_axes("bottom", size="5%", pad=0.01)
# cbar = fig.colorbar(im, im,fraction=0.046, pad=0.04) #cax = cax, orientation='horizontal')
# cbar.ax.set_xlabel('E Field Magnitude')      
# plt.show()
