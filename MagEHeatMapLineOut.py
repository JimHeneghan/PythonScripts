#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

fig = plt.figure(1, figsize = (7, 8), constrained_layout=True)
ax = fig.add_subplot(111)
Frame = []
Nx = 119
Ny = 207
z = 0
dt = 5.945e-18
T = 2e6

shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
freq = ['6.1', '6.3', '6.5', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
for z in range(0, 7):
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
		print(min(Full[i]))

	ax.plot(X, Full[103] + (5e-21)*(z), linewidth=2, color = shades[len(shades)-z - 1], label = r"$ \rm %s \ \mu m$" %freq[z])



	# ax0.set_title(r"$\rm \lambda_{|E|} = %s \ \mu m $" %lam, fontsize = '25')

	# ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	# ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	# plt.setp(ax0.spines.values(), linewidth=2)
	# plt.tick_params(left = False, bottom = False)
	# z = z + 1

# cbar = fig.colorbar(im, ax=axs)
# cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')
# plt.savefig("EabsTest3.png")

ax.legend(loc = 'upper left',  ncol = 1, fancybox= True, framealpha = 0.5, shadow = False, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
# ax.set_xlim(0.3, 4.1)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(left = False, bottom = False,labelsize = 'large')
ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax.set_ylabel(r"$\rm \vert E \vert \ Field \ (V m^{-1})$", fontsize = '35')
plt.savefig("2_38HMLineOutsHalfNov18Vert.png")
plt.savefig("2_38HMLineOutsHalfNov18Vert.pdf")

# plt.savefig("FullDev2_42um|E|_BarSource6_7.pdf")
# plt.savefig("FullDev2_42um|E|_BarSource6_7.png")
#abs(Full*(dt/T))
