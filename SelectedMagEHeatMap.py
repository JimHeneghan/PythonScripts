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

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
pitch = [6.3, 6.5, 6.7, 6.9, 7.0, 7.1, 7.2, 7.5, 7.6, 7.9]
Nx = 119
Ny = 207
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
for ax0 in axs.flat:
	lam = pitch[z]
	Fieldx = "ExHeatMap%s.txt" %lam 
	Fieldy = "EyHeatMap%s.txt" %lam 
	Fieldz = "EzHeatMap%s.txt" %lam 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	ExAb = np.zeros((Ny, Nx), dtype = np.double)
	EyAb = np.zeros((Ny, Nx), dtype = np.double)
	EzAb = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny/100, Ny)
	X = np.linspace(0,  Nx/100, Nx)

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

	# print(max(np.sqrt(Ex)))
	
	norm = mpl.colors.Normalize(vmin=0, vmax=5e-21) 
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{|E|} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("EModSelectedVals2_38um.png")

# plt.savefig("FullDev2_42um|E|_BarSource6_7.pdf")
# plt.savefig("FullDev2_42um|E|_BarSource6_7.png")
#abs(Full*(dt/T))


#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

# fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
# Frame = []
# Nx = 119
# Ny = 207
# z = 0
# dt = 5.945e-18
# T = 2e6
# #print(z)
# for ax0 in axs.flat:
# 	lam = 7.0 + 0.1*z
# 	Fieldx = "ExHeatMap%s.txt" %lam 
# 	Fieldy = "EyHeatMap%s.txt" %lam 
# 	Fieldz = "EzHeatMap%s.txt" %lam 

# 	Full = np.zeros((Ny, Nx), dtype = np.double)
# 	ExAb = np.zeros((Ny, Nx), dtype = np.double)
# 	EyAb = np.zeros((Ny, Nx), dtype = np.double)
# 	EzAb = np.zeros((Ny, Nx), dtype = np.double)

# 	Y = np.linspace(0,  Ny, Ny)
# 	X = np.linspace(0,  Nx, Nx)

# 	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
# 	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
# 	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )

# 	Ex = Ex*Ex*(dt/T)*(dt/T)
# 	Ey = Ey*Ey*(dt/T)*(dt/T)
# 	Ez = Ez*Ez*(dt/T)*(dt/T)

# 	for i in range (0, Ny):
# 		Full[i] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
# 		Full[i] = np.sqrt(Full[i])
# 		# print(max(Full[i]))

# 	# print(max(np.sqrt(Ex)))
	
# 	norm = mpl.colors.Normalize(vmin=0, vmax=5e-21) 
# 	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

# 	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

# 	ax0.set_title(r"$\rm \lambda_{|E|} = %s \ \mu m $" %lam, fontsize = '25')

# 	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
# 	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
# 	plt.setp(ax0.spines.values(), linewidth=2)
# 	plt.tick_params(left = False, bottom = False)
# 	z = z + 1

# cbar = fig.colorbar(im, ax=axs)
# cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')
# # plt.savefig("EabsTest2.png")

# plt.savefig("FullDev2_42um|E|_BarSource7_8.pdf")
# plt.savefig("FullDev2_42um|E|_BarSource7_8.png")