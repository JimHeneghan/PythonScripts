#!/usr/local/apps/anaconda/3-2.2.0/bin/python
import time 
import numpy as np
import scipy as sp

import itertools 
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl


shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
freq = ['6.1', '6.3', '6.4', '6.5', '6.9', '7.0']
c0 = 3e8
threshold = 0.02
numpad = 100000
Eline = {}
lam1  = {}
lam2  = {}
lam3  = {}
lam4  = {}
MyLegend = []
peakx = []
peakE = []

peaknegx = []
peaknegE = []
# for z in range(6, 0):
z = 4
q = 0

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

# while (z > 3):#(-1)):
fig = plt.figure(1, figsize = (12, 8), constrained_layout=True)
ax1 = fig.add_subplot(111)


#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

Frame = []
Nx = 119
Ny = 207
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
freq = ['6.1', '6.3', '6.5', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
for z in range(3, 4):
	print z
	Field = "EyHeatMap6.8.txt" 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	Line = np.zeros(Nx, dtype = np.double)

	Y = np.linspace(-2.13, 2.13, Ny)
	X = np.linspace(-1.23, 1.23, Nx)
	x = np.linspace(-1.7395,  1.7395, Nx)

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
	E = Line
	for i in range(0,len(E)-1):
	    if ((E[i] > E[i-1]) & (E[i] > E[i+1])):	        
	    	peakx.append(x[i])
	    	peakE.append(E[i])
	for i in range(0,len(E)-1):
	    if ((E[i] < E[i-1]) & (E[i] < E[i+1])):	        
	    	peaknegx.append(x[i])
	    	peaknegE.append(E[i])
			# print(min(Full[i]))

	handles, labels = ax1.get_legend_handles_labels()




Eline[q], = ax1.plot(x, Line, linewidth=2, color = 'black')
dist = abs(peakx[1] - peakx[0])
ax1.scatter(peakx, peakE, s=25,edgecolors = 'black', c='blue', zorder = 25, label = "Posative Peaks")		
ax1.scatter(peaknegx, peaknegE, s=25,edgecolors = 'black', c='red', zorder = 25, label = "Negative Peaks")		

print("Positive peaks")
for i in range(0, len(peakx)):
	print("%f \t %f" %(peakx[i], peakE[i]))

print("\n \n")

print("Negative peaks")
for i in range(0, len(peaknegx)):
	print("%f \t %f" %(peaknegx[i], peaknegE[i]))

ax1.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax1.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax1.axhline(y =  0.0, linestyle = "dashed", color = 'black')

ax1.set_ylabel(r"$\rm E_{y} \ Field \ V m^{-1}$", fontsize = '20')
ax1.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
plt.setp(ax1.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax1.legend(loc = 'upper right',  ncol = 1, fancybox= True, framealpha = 0.5, shadow = False, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),

plt.savefig("6_8DiagLineOutPeak2.pdf")
plt.savefig("6_8DiagLineOutPeak2.png")
#abs(Full*(dt/T))

