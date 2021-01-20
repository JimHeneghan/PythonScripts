import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(1, figsize = (7, 8), constrained_layout=True)
ax = fig.add_subplot(111)
cmatch = ["c", "olive", "purple", "goldenrod", "orangered", "steelblue"]
freq = ['6.1', '6.3', '6.4', '6.5', '6.9', '7.0']
for z in range(0, len(freq)):
	print z
	Field = "%sPlane.txt" %freq[z]
	# z = z
	Full = np.zeros((221, 129), dtype = np.double)
	X = np.linspace(0,  2.281, 129)
	Y = np.linspace(0,  4.01, 221)
	
	for i in range (0, 221):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 357, unpack =True )
		Full[i] = E 

	# pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
	# fig, ax0 = plt.subplots(figsize = (3.5, 3.5),constrained_layout=True)
	# ax0.set_aspect(401/231)
	# norm = mpl.colors.Normalize(vmin=0, vmax=3.5)
	# im = ax0.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)
	
	# cbar = fig.colorbar(im, ax=ax0)
	# cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')

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
	ax.plot(X[8:121] - 1.15, Full[110][8:121] + 1.8*(z), linewidth=2, color = cmatch[len(cmatch)-z - 1], label = r"$ \rm %s \ \mu m$" %freq[z])

	#plt.savefig("EFieldXSec43THz.pdf")
ax.legend(loc = 'upper right',  ncol = 1, fancybox= True, shadow = True, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
# ax.set_ylim(0, 14)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(left = False, bottom = False)
ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax.set_ylabel(r"$\rm \|E\|  \ (V m^{-1})$", fontsize = '35')
plt.savefig("EFieldXSec_Line_AbsE_Line1_WLegend_June10.png")
plt.savefig("EFieldXSec_Line_AbsE_Line1_WLegend_June10.pdf")
# plt.show()

