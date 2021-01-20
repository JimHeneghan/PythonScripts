#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

fig = plt.figure(1, figsize = (9, 8), constrained_layout=True)
ax1 = fig.add_subplot(111)
Frame = []
Nx = 119
Ny = 207
z = 0
dt = 5.945e-18
T = 2e6
E = []
peakx = []
peakE = []
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
freq = ['6.1', '6.3', '6.5', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
for z in range(3, 4):
	print z
	# Field = "%s_um.txt" %freq[z]
	Fieldx = "ExHeatMap%s.txt" %freq[z] 
	Fieldy = "EyHeatMap%s.txt" %freq[z]
	Fieldz = "EzHeatMap%s.txt" %freq[z] 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	ExAb = np.zeros((Ny, Nx), dtype = np.double)
	EyAb = np.zeros((Ny, Nx), dtype = np.double)
	EzAb = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(-1.19,  1.19, Nx)

	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )

	# Ex = Ex*Ex*(dt/T)*(dt/T)
	# Ey = Ey*Ey*(dt/T)*(dt/T)
	# Ez = Ez*Ez*(dt/T)*(dt/T)

	Ex = abs(Ex)*abs(Ex)*(dt/T)*(dt/T)
	Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
	Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

	for i in range (0, Ny):
		Full[i] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
		Full[i] = np.sqrt(Full[i])
		# print(min(Full[i]))
	E = 1e20*Full[103]
	print(E)

	for i in range(0,len(E)-1):
	    if ((E[i] > E[i-1]) & (E[i] > E[i+1]) & (E[i] > 0.45)):	        
        	peakx.append(X[i])
        	peakE.append(E[i])
	# ax0.set_title(r"$\rm \lambda_{|E|} = %s \ \mu m $" %lam, fontsize = '25')
	print(peakx)
	print(peakE)
	# ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	# ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	# plt.setp(ax0.spines.values(), linewidth=2)
	# plt.tick_params(left = False, bottom = False)
	# z = z + 1

# cbar = fig.colorbar(im, ax=axs)
# cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')
# plt.savefig("EabsTest3.png")



ax = ax1.twinx()

x, E = np.loadtxt("../6.9_um.txt", usecols=(0,1), skiprows= 3, unpack =True )
x = x - max(x)/2
ax1.plot(x*1e6, E, color = 'black', zorder=30, linewidth = 4, label = "s-SNOM Data")
ax1.legend(loc = 'upper right',  ncol = 1, fancybox= True, framealpha = 0.5, shadow = False, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),

ax.plot(X, 1e20*Full[103], linewidth=2, color = 'red', zorder = 10, label = "FDTD Lineout")
dist = abs(peakx[1] - peakx[0])
ax.scatter(peakx, peakE, s=12,edgecolors = 'black', c='red', zorder = 25, label = r"$ peak \ dist = %2.2f \ \mu m$" %dist )		

ax.legend(loc = 'upper left',  ncol = 1, fancybox= True, framealpha = 0.5, shadow = False, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax.set_xlim(-1.15, 1.15)
ax1.set_xlim(-1.15, 1.15)

ax.xaxis.set_minor_locator(AutoMinorLocator(10))

ax.tick_params(which = 'major', direction = 'in', width=2, length = 7, labelsize=20)
ax.tick_params(which = 'minor', direction = 'in', width=2, length = 5, labelsize=20)

# ax.xaxis.grid(True, which='minor')

plt.setp(ax.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False,labelsize = 'large')
ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax.set_ylabel(r"$\rm \vert E \vert \ Field \  (arb. units)$", color = 'red', fontsize = '20')

ax1.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. units)$", fontsize = '20')

plt.savefig("6_9LineoutsComp.png")
plt.savefig("6_9LineoutsComp.pdf")

# plt.savefig("FullDev2_42um|E|_BarSource6_7.pdf")
# plt.savefig("FullDev2_42um|E|_BarSource6_7.png")
#abs(Full*(dt/T))
