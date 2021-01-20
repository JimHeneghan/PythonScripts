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
############################## Ex Field ################################
#______________________________________________________________________#
fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
for ax0 in axs.flat:
	lam = 6.0 + 0.1*z
	Field = "ExHeatMap%s.txt" %lam 
	# print(Field)
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	# Er = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	# Ei = np.loadtxt(Field, usecols=(1,), skiprows= 1, unpack =True )
	# for i in range (0, Ny):
	#  	Full[i] = (Er[i*Nx: i*Nx + Nx]*(dt/T) + 1j*Ei[i*Nx: i*Nx + Nx]*(dt/T))*(Er[i*Nx: i*Nx + Nx]*(dt/T) - 1j*Ei[i*Nx: i*Nx + Nx]*(dt/T))


	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	# print(max((Er[i*Nx: i*Nx + Nx]*(dt/T) + 1j*Ei[i*Nx: i*Nx + Nx]*(dt/T))*(Er[i*Nx: i*Nx + Nx]*(dt/T) - 1j*Ei[i*Nx: i*Nx + Nx]*(dt/T))))
	# print(min((Er[i*Nx: i*Nx + Nx]*(dt/T) + 1j*Ei[i*Nx: i*Nx + Nx]*(dt/T))*(Er[i*Nx: i*Nx + Nx]*(dt/T) - 1j*Ei[i*Nx: i*Nx + Nx]*(dt/T))))

	# print(min(E*(dt/T)))
	print(max(E*(dt/T)))
	norm = mpl.colors.Normalize(vmin=-2e-21, vmax=-2e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	# ax0.set_aspect(4/2.3)

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %lam, fontsize = '25')

	# ax0.axhline(y =115, color = 'fuchsia', linewidth=1)
	# ax0.axhline(y =415, color = 'fuchsia', linewidth=1)

	# ax0.axhline(y =265, color = 'silver', linewidth=1)
	# ax0.axhline(y =290, color = 'silver', linewidth=1)

	# ax0.axhline(y =125, color = 'yellow', linewidth=1)



	#ax0.set_title(r"$ \rm  %.2f \ (\mu m)$" % fontsize = '25')
	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

	# plt.show()
cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38umEx_Bar_Complex.pdf")
plt.savefig("FullDev2_38umEx_Bar_Complex.png")

print("\n \n")


#______________________________________________________________________#
############################## Ey Field ################################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 6.0 + 0.1*z
	Field = "EyHeatMap%s.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-2e-21, vmax=2e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ey} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ey \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38umEy_BarSource.pdf")
plt.savefig("FullDev2_38umEy_BarSource.png")

print("\n \n")
#______________________________________________________________________#
############################## Ex Again ################################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 6.0 + 0.1*z
	Field = "ExHeatMap%s.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-2e-21, vmax=2e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38umEx2_BarSource.pdf")
plt.savefig("FullDev2_38umEx2_BarSource.png")

print("\n \n")
#______________________________________________________________________#
############################## Ez Field ################################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 6.0 + 0.1*z
	Field = "EzHeatMap%s.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))

	norm = mpl.colors.Normalize(vmin=-1e-21, vmax=1e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ez} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ez \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38umEz_BarSource.pdf")
plt.savefig("FullDev2_38umEz_BarSource.png")
print("\n \n")

#______________________________________________________________________#
############################## Ey Field 7_8 um #########################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 7.0 + 0.1*z
	Field = "EyHeatMap%s.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-2e-21, vmax=2e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ey} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ey \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38umEy_BarSource7_8.pdf")
plt.savefig("FullDev2_38umEy_BarSource7_8.png")

print("\n \n")
#______________________________________________________________________#
############################## Ex Again 7-8 um #########################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 7.0 + 0.1*z
	Field = "ExHeatMap%s.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))
	norm = mpl.colors.Normalize(vmin=-2e-21, vmax=2e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ex \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38umEx2_BarSource7_8.pdf")
plt.savefig("FullDev2_38umEx2_BarSource7_8.png")

print("\n \n")
#______________________________________________________________________#
############################## Ez Field 7-8 um #########################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 7.0 + 0.1*z
	Field = "EzHeatMap%s.txt" %lam 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	print(max(E)*(dt/T))

	norm = mpl.colors.Normalize(vmin=-1e-21, vmax=1e-21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ez} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm Ez \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38umEz_BarSource7_8.pdf")
plt.savefig("FullDev2_38umEz_BarSource7_8.png")
print("\n \n")

#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

fig, axs = plt.subplots(2, 5, figsize = (18, 9),constrained_layout=True)
Frame = []
Nx = 119
Ny = 207
z = 0
#print(z)
for ax0 in axs.flat:
	lam = 6.0 + 0.1*z
	Fieldx = "ExHeatMap%s.txt" %lam 
	Fieldy = "EyHeatMap%s.txt" %lam 
	Fieldz = "EzHeatMap%s.txt" %lam 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	ExAb = np.zeros((Ny, Nx), dtype = np.double)
	EyAb = np.zeros((Ny, Nx), dtype = np.double)
	EzAb = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(0,  Nx, Nx)

	Ex = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	Ey = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	Ez = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
		ExAb[i] = Ex[i*Nx: i*Nx + Nx]**2
		EyAb[i] = Ey[i*Nx: i*Nx + Nx]**2
		EzAb[i] = Ez[i*Nx: i*Nx + Nx]**2
		Full[i] = np.sqrt(ExAb[i]+ EyAb[i] + EzAb[i])

	print(max(E)*(dt/T))
	
	
	norm = mpl.colors.Normalize(vmin=0, vmax=1e21)
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = ax0.pcolormesh(X, Y, abs(Full*(dt/T)), norm=norm, **pc_kwargs)

	ax0.set_title(r"$\rm \lambda_{Ex} = %s \ \mu m $" %lam, fontsize = '25')

	ax0.set_xlabel(r"$ \rm x \ $", fontsize = '20')
	ax0.set_ylabel(r"$ \rm y \ $", fontsize = '20')
	plt.setp(ax0.spines.values(), linewidth=2)
	plt.tick_params(left = False, bottom = False)
	z = z + 1

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm |E| \ \ Field \ (V m^{-1})$", size = '20')
plt.savefig("FullDev2_38um|E|_BarSource2.pdf")
plt.savefig("FullDev2_38um|E|_BarSource2.png")
