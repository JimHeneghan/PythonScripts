import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
rain = ['red', 'orange', 'yellow', 'limegreen', 'cyan']
xint = [61, 101, 129, 169]
for z in range(44, 45):
	Field = "%dHzXSec.txt" %z
	Full = np.zeros((601, 231), dtype = np.double)

	X = np.linspace(0,  2.31, 231)
	Y = np.linspace(0,  0.6, 601)
	ycon = []
	for i in range (0, 231):
		ycon.append(0.375)
	for i in range (0, 600):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 839, unpack =True )
		Full[i] = E**2 


	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	fig, ax0 = plt.subplots(figsize = (8, 3.08),constrained_layout=True)
	#ax0.set_aspect(231/601)
	norm = mpl.colors.Normalize(vmin=0, vmax=10)
	im = ax0.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)
	
	cbar = fig.colorbar(im, ax=ax0)
	cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

	lam = (3e8/(z*1e12))*1e6 
	#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" %lam, fontsize = '25')
	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')


	for i in range (1, len(xint)):
		ax0.plot(X[xint[i-1]:xint[i]], ycon[xint[i-1]:xint[i]], linewidth = 4, color = rain[i-1])
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)

	#plt.savefig("XSec2.3PitchBN_Ag_Si%dHz.pdf" %z)
	#plt.savefig("XSec2.3PitchBN_Ag_Si%dHz.png" %z)
	#plt.show()

	# fig, ax1 = plt.subplots(figsize = (6, 3.5),constrained_layout=True)
	# ax1.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	# ax1.set_ylabel(r"$ \rm E$", fontsize = '20')
	# plt.setp(ax1.spines.values(), linewidth=2)
	# plt.tick_params(left = False, bottom = False)
	# # plt.xlim(0.1, 2)
	# plt.plot(X[15:200], Full[200][15:200], color = "black", linewidth=3)
	plt.savefig("IntXSec44THzLine.png")
	plt.savefig("IntXSec44THzLine.pdf")
	# plt.show()

