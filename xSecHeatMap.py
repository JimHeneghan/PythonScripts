import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable

for z in range(44, 45):
	Field = "%dHzXSec.txt" %z
	Full = np.zeros((601, 253), dtype = np.double)

	X = np.linspace(0,  2.51, 253)
	Y = np.linspace(0,  0.383, 601)
	for i in range (0, 600):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 861, unpack =True )
		Full[i] = E 

	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	fig, ax0 = plt.subplots(figsize = (10.04, 4),constrained_layout=True)
	#ax0.set_aspect(231/601)
	norm = mpl.colors.Normalize(vmin=0, vmax=3.5)
	im = ax0.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)
	
	cbar = fig.colorbar(im, ax=ax0)
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

	lam = (3e8/(z*1e12))*1e6 
	#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" %lam, fontsize = '25')
	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')

	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)

	plt.savefig("XSec2.51PitchBN_Ag_Si%dHz.pdf" %z)
	plt.savefig("XSec2.51PitchBN_Ag_Si%dHz.png" %z)
	#plt.show()

	# fig, ax1 = plt.subplots(figsize = (6, 3.5),constrained_layout=True)
	# ax1.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	# ax1.set_ylabel(r"$ \rm E$", fontsize = '20')
	# plt.setp(ax1.spines.values(), linewidth=2)
	# plt.tick_params(left = False, bottom = False)
	# # plt.xlim(0.1, 2)
	# plt.plot(X[15:200], Full[200][15:200], color = "black", linewidth=3)
	# plt.savefig("EFieldXSec43THz.png")
	# plt.savefig("EFieldXSec43THz.pdf")
	plt.show()

