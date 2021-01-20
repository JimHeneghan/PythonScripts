import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

for z in range(33, 51):
	Field = "%dHz.txt" %z
	Full = np.zeros((401, 231), dtype = np.double)

	X = np.linspace(0,  2.31, 231)
	Y = np.linspace(0,  4.01, 401)
	for i in range (0, 400):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 639, unpack =True )
		Full[i] = E 

	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	fig, ax0 = plt.subplots(figsize = (4, 4),constrained_layout=True)
	ax0.set_aspect(401/231)
	im = ax0.pcolormesh(X, Y, Full, **pc_kwargs)
	fig.colorbar(im, ax=ax0, shrink = 0.6, label = r"$\rm E \ \ Field\ \ V m^{-1}$")

	ax0.set_title('%d Hz' %z)
	ax0.set_xlabel(r"$ \mu \rm m$")
	ax0.set_ylabel(r"$ \mu \rm m$")

	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)

	# plt.savefig("hBN_Ag_Si%dHz.pdf" %z)
	# plt.savefig("hBN_Ag_Si%dHz.png" %z)
plt.show()

