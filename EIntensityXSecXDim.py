import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(1, figsize = (3.86, 5.55), constrained_layout=True)
ax = fig.add_subplot(111)
cmatch = ["black", "red", "blue", "fuchsia", "green", "navy", "blueviolet", "darkmagenta", "maroon", "yellowgreen", "steelblue" ]
for z in range(38, 49):
	Field = "%dHz.txt" %z
	z = z
	Full = np.zeros((401, 231), dtype = np.double)

	X = np.linspace(0,  2.31, 231)
	Y = np.linspace(0,  4.01, 401)
	for i in range (0, 400):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 639, unpack =True )
		Full[i] = E 

	# pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	# fig, ax0 = plt.subplots(figsize = (3.5, 3.5),constrained_layout=True)
	# ax0.set_aspect(401/231)
	# norm = mpl.colors.Normalize(vmin=0, vmax=3.5)
	# im = ax0.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)
	
	# cbar = fig.colorbar(im, ax=ax0)
	# cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

	lam = (3e8/(z*1e12))*1e6 
	# ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" %lam, fontsize = '25')
	# ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	# ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')

	# plt.setp(ax0.spines.values(), linewidth=2)
	# plt.tick_params(left = False, bottom = False)

	#plt.savefig("hBN_Ag_Si%dHz.pdf" %z)
	#plt.savefig("hBN_Ag_Si%dHz.png" %z)
	#plt.show()

	#fig, ax1 = plt.subplots(figsize = (6, 3.5),constrained_layout=True)
	#plt.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	#plt.set_ylabel(r"$ \rm E$", fontsize = '20')
	#plt.setp(spines.values(), linewidth=2)
	#plt.tick_params(left = False, bottom = False)
	# plt.xlim(0.1, 2)
	#print X[0:15]
	ax.plot(X[15:215] - 1.15, Full[200][15:215]**2 -2.2*(z - 38), color = cmatch[len(cmatch) - (z - 37)], linewidth=2,  label = r"$ \rm %.2f \ (\mu m)$" %lam)
	#plt.savefig("EFieldXSec43THz.png")
	#plt.savefig("EFieldXSec43THz.pdf")
ax.legend(loc = 'lower center', bbox_to_anchor=(1.05, 0.5), ncol = 1, fancybox= True, shadow = True, fontsize = 12)
ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)
ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
#plt.savefig("/halhome/jimheneghan/UsefulImages/EIntensityXSecMar15NoLeg.png")
plt.show()

