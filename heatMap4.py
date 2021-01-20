import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
freq = [33, 35, 37, 39, 41, 43, 44, 45, 47, 49]
many = np.zeros((10, 401, 253), dtype = np.double)
j = 0
for z in range(0, 10):
	Field = "%dHz.txt" %freq[z]
	Full = np.zeros((401, 253), dtype = np.double)

	X = np.linspace(0,  2.51, 253)
	Y = np.linspace(0,  4.01, 401)
	for i in range (0, 400):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 697, unpack =True )
		Full[i] = E 

	many[j] = Full
	j = j + 1

pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
fig, axs = plt.subplots(2, 5, figsize = (16, 8),constrained_layout=True)
q = 33
i = 0
for ax in axs.flat:
	
	ax.set_aspect(401/253)
	norm = mpl.colors.Normalize(vmin=0, vmax=1.5)
	im = ax.pcolormesh(X, Y, many[i], norm = norm, **pc_kwargs)
		
	
	lam = (3e8/(freq[i]*1e12))*1e6 
	i = i + 1
	ax.set_title(r"$ \rm  %.2f \ (\mu m)$" %lam, fontsize = '25')
	ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')

	plt.setp(ax.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm \|E\| \ Field \ (V m^{-1})$", size = '20')

plt.savefig("2.51PitchSelfNorm.pdf")
plt.savefig("2.51PitchSelfNorm.png")
plt.show()
