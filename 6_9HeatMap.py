import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable


Field = "6.9um.txt"
Full = np.zeros((359, 209), dtype = np.double)

X = np.linspace(0,  2.31, 209)
Y = np.linspace(0,  4.01, 359)
for i in range (0, 359):
	E = np.loadtxt(Field, usecols=(i,), skiprows=575, unpack =True )
	Full[i] = E 

# pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
# fig, ax0 = plt.subplots(figsize = (3.5, 3.5),constrained_layout=True)
# ax0.set_aspect(401/231)
# norm = mpl.colors.Normalize(vmin=0, vmax=3)
# im = ax0.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)

# cbar = fig.colorbar(im, ax=ax0)
# cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')

# lam = 6.5 
# ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" %lam, fontsize = '25')
# ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
# ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')

# plt.setp(ax0.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)

# plt.savefig("hBN_Ag_Si6.5um.pdf")
#plt.savefig("hBN_Ag_Si6.5um.png")
#plt.show()

fig, ax1 = plt.subplots(figsize = (6, 3.5),constrained_layout=True)
ax1.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
ax1.set_ylabel(r"$ \rm |E|$", fontsize = '20')
plt.setp(ax1.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)
# plt.xlim(0.1, 2)
plt.plot(X[4:204] - 1.15, Full[200][4:204], color = "black", linewidth=2)
ax1.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax1.axvline(x =  0.68, linestyle = "dashed", color = 'black')
plt.savefig("6.9umSourceXSecLineMar15.png")
plt.savefig("6.9umSourceXSecLineMar15.pdf")
plt.show()

