#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar, SI_LENGTH

#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

fig, ((ax1, ax0), (ax2, ax3)) = plt.subplots(2,2, figsize = (16,15))#, constrained_layout=True)


Frame = []
Nx = 123
Ny = 213
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
shades = ['black', 'red', 'orange', 'yellow', 'lime', 'green', 'springgreen', 'teal', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']

freq = ['6.1', '6.3', '6.5', '6.7', '6.8', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
selectionBoxAli = [1, 2, 5, 6, 9]
selectionBoxJim = [1, 2, 3, 4, 5, 6, 9]

for z in range(3, 4):
	print z
	Field = "EyHeatMap6.8.txt" 
	
	Full = np.zeros((Ny, Nx), dtype = np.double)
	Line = np.zeros(Nx, dtype = np.double)
	Linx = np.zeros(Nx, dtype = np.double)
	Liny = np.zeros(Nx, dtype = np.double)


	Y = np.linspace(-2.13, 2.13, Ny)
	X = np.linspace(-1.23, 1.23, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)

	k = 0
	xstart = 0
	ystart = 168
	for i in range(0, Nx):
		Line[k] = Full[ystart, xstart]
		Linx[k] = (xstart - 61)/50.0
		Liny[k] = (ystart - 106)/50.0
		k = k+1
		xstart = xstart + 1
		ystart = ystart - 1	

print(Linx)
print(Liny)
# ax0.set_title(r"$\rm |E| \ \lambda = %s \ \mu m $" %freq[z], pad=20, fontsize = '35')
	
norm = mpl.colors.Normalize(vmin=-1, vmax=1) 
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

im = ax0.pcolormesh(X, Y, 1e21*Full, norm=norm, **pc_kwargs)
ax0.set_aspect('equal')
ax0.scatter(Linx, Liny, linewidth = 3, s=15,edgecolors = 'black', c='silver', zorder = 25, label = r"$Line \ Out \ Points$")  

ax0.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '25')
ax0.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '25')
plt.setp(ax0.spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)

cbar = fig.colorbar(im, ax=ax0)
cbar.set_label(label = r"$\rm E_{y} \ Field \ (arb. units)$", size = '20')


#####################################################################3
Nx = 125
Ny = 250

for z in range(4, 5):
	print z
	# Field = "%s_um.txt" %freq[z]
	Fieldx = "Ali6.9_um.txt" 
	# Fieldy = "Ali%s_um.txt" %freq[z]
	# Fieldz = "Ali%s_um.txt" %freq[z] 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	Ex = np.zeros((Ny, Nx), dtype = np.double)
	# Ey = np.zeros((Ny, Nx), dtype = np.double)
	# Ez = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  10, Ny)
	X = np.linspace(0,  5, Nx)

	Full = np.loadtxt(Fieldx,  skiprows= 0, unpack =True )
	# Full = np.reshape(Full, (Ny, Nx), order='C')

	# Ex = Ex*Ex*(dt/T)*(dt/T)
	# Ey = Ey*Ey*(dt/T)*(dt/T)
	# Ez = Ez*Ez*(dt/T)*(dt/T)

	# Ex = abs(Ex)*abs(Ex)*(dt/T)*(dt/T)
	# Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
	# Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

	# for i in range (0, Ny):
	# 	Full[i] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
	# 	Full[i] = np.sqrt(Full[i])
	# 	print(i)
	# print(min(Full[i]))

	# print(max(np.sqrt(Ex)))
# ax1.set_title(r"$\rm \lambda_{|E|} = %s \ \mu m $" %freq[z], pad=20, fontsize = '35')
# scalebar = ScaleBar(0.08, units ='um', dimension = SI_LENGTH)	
# ax1.add_artist(scalebar)

norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4) 
pc_kwargs = {'rasterized': True, 'cmap':'jet'}


im = ax1.pcolormesh(X, Y, Full, norm=norm, **pc_kwargs)
ax1.set_aspect('equal', anchor = (0.1, 0.5), adjustable = 'box')

ax1.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '25')
ax1.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '25')
# ax1.set_yticklabels([])
# ax1.set_xticklabels([])
# ax1.set_yticks([])
# ax1.set_xticks([])

plt.setp(ax1.spines.values(), linewidth=2)
# ax1.tick_params(left = False, bottom = False)
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label(label = r"$\rm s-SNOM \ Signal \ (arb. units)$", size = '25')

for z in range(0, 5):
	print z
	maxSel = len(selectionBoxAli) - 1
	sbox = selectionBoxAli[maxSel - z]
	Field = "../%s_um.txt" %freq[sbox]
	# z = z
	x, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )
	x = x - max(x)/2

	# if (z == 2):
	# 	E = E + 0.15
	mult = 0.4
	if (z > 1):
		mult = 0.25

	if (z == 4):
		mult = 0.2

	ax2.plot(x*1e6, E - mult*(z), linewidth=2, color = shades[sbox], label = r"$ \rm %s \ \mu m$" %freq[sbox])

	#plt.savefig("EFieldXSec43THz.pdf")
ax2.legend(loc = 'upper left',  ncol = 1, fancybox= True, framealpha = 0.5, shadow = False, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
ax2.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax2.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax2.set_xlim(-1.15, 1.15)
# plt.setp(ax2.spines.values(), linewidth=1)
ax2.tick_params(left = False, bottom = False,labelsize = 'large')
ax2.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax2.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. units)$", fontsize = '35')

Nx = 123
Ny = 213
###############################################################

for z in range(0, 7):
	print z
	maxSel = len(selectionBoxJim) - 1

	sbox = selectionBoxJim[maxSel - z]
	print(sbox)
	print(freq[sbox])
	print ("\n")
	# Field = "%s_um.txt" %freq[z]
	Field = "EyHeatMap%s.txt" %freq[sbox]

	Full = np.zeros((Ny, Nx), dtype = np.double)
	Line = np.zeros(Nx, dtype = np.double)

	Y = np.linspace(-2.13, 2.13, Ny)
	X = np.linspace(-1.23, 1.23, Nx)

	E = np.loadtxt(Field, usecols=(0,), skiprows= 1, unpack =True )
	for i in range (0, Ny):
	 	Full[i] = E[i*Nx: i*Nx + Nx]*(dt/T)
		# print(min(Full[i]))

	k = 0
	xstart = 0
	ystart = 168
	for i in range(0, Nx):
		Line[k] = Full[ystart, xstart]
		k = k+1
		xstart = xstart + 1
		ystart = ystart - 1

	if ((sbox==3) or (sbox==4)):
		ax3.plot(X, 1e20*Line - (0.4)*(z), linewidth=6, color = 'black')
	ax3.plot(X, 1e20*Line - (0.4)*(z), linewidth=2, color = shades[sbox], label = r"$ \rm %s \ \mu m$" %freq[sbox])



ax3.legend(loc = 'upper left',  ncol = 1, fancybox= True, framealpha = 0.5, shadow = False, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
ax3.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax3.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax3.set_xlim(-1.23, 1.23)
# plt.setp(ax3.spines.values(), linewidth=1)
ax3.tick_params(left = False, bottom = False,labelsize = 'large')
ax3.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax3.set_ylabel(r"$\rm E_{y} \ Field  \ (arb. units)$", fontsize = '35')

plt.tight_layout()
plt.savefig("4PanelDiagLineDots.png")
plt.savefig("4PanelDiagLineDots.pdf")