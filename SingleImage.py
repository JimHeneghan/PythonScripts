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
	Field = "RealFullSpace%dHz.txt" %z
	Full = np.zeros((401, 231), dtype = np.double)

	X = np.linspace(0,  2.31, 231)
	Y = np.linspace(0,  4.01, 401)
	for i in range (0, 401):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 639, unpack =True )
		Full[i] = E 

	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	fig, ax0 = plt.subplots(figsize = (3.5, 3.5),constrained_layout=True)
	ax0.set_aspect(401/231)
	norm = mpl.colors.Normalize(vmin=0, vmax=3.5)
	im = ax0.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)
	
	# cbar = fig.colorbar(im, ax=ax0)
	# cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

	lam = (3e8/(z*1e12))*1e6 
	# ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" %lam, fontsize = '25')
	# ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	# ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')
	plt.xticks([])
	plt.yticks([])
	# plt.setp(ax0.spines.values(), linewidth=2)
	# plt.tick_params(left = False, bottom = False)

	#plt.savefig("hBN_Ag_Si%dHz.pdf" %z)
	#plt.savefig("hBN_Ag_Si%dHz.png" %z)
	#plt.show()

	# fig, ax1 = plt.subplots(figsize = (6, 3.5),constrained_layout=True)
	# ax1.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	# ax1.set_ylabel(r"$ \rm E$", fontsize = '20')
	# plt.setp(ax1.spines.values(), linewidth=2)
	# plt.tick_params(left = False, bottom = False)
	# # plt.xlim(0.1, 2)
	# plt.plot(X[15:200], Full[200][15:200], color = "black", linewidth=3)
	plt.savefig("FullSpace44THzMar6.png")
	plt.savefig("FullSpace44THzMar6.pdf")
	plt.show()

