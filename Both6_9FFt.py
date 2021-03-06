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
# for z in range(6, 0):
z = 4
q = 0

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

# while (z > 3):#(-1)):
fig = plt.figure(1, figsize = (12, 8), constrained_layout=True)
ax1 = fig.add_subplot(111)
# 	print(freq[z])
# 	Field = "../%s_um.txt" %freq[z]
# 	# z = z
# 	x, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )

# 	padE = np.pad(E, numpad, mode='constant')
# 	# Efft = np.fft.fft(padE, len(padE))
# 	# fs = 1/(x[1]*len(padE))
# 	# f = np.arange(0, fs*len(padE), fs)
# 	# # lam = 1/f
# 	n = padE.size 
# 	dx = x[1]
# 	dk = 1/(n*dx)
# 	k = np.arange(0,n*dk, dk)


# 	Efft = abs(np.fft.fft(padE))

# 	# Fourier transform data and take absolute value
# 	# Efft = abs(fft(pe))

# 	# print("Excitation Frequency", 1/(lam/1e4), "cm^-1")

# 	# Find peaks and store peak data
# 	counter = 0
# 	lam = []
# 	peakk = []
# 	peakEfft = []
# 	for i in range(2,n//2):
# 	    if ((Efft[i] > Efft[i-1]) & (Efft[i] > Efft[i+1]) & (Efft[i] > threshold)&(counter < 5)):
# 	        print(Efft[i], 1.0/k[i], "um", 2*math.pi*k[i]*1e4, "cm-1")
# 	        peakk.append(k[i])
# 	        peakEfft.append(Efft[i])
# 	        # print(counter)
# 	        if ((counter < 5)&(counter > 0)):
# 	        	lam.append(1e6/k[i])
# 	        	# print(lam[counter - 1])
#         	counter = counter + 1

# 	# ax.plot(k, Efft, linewidth=2, label = r"$ \rm %s \ \mu m , \ \lambda_{exciton} = %2.2f \ \mu m , \  %2.2f \ \mu m , \  %2.2f \ \mu m , \  %2.2f \ \mu m  $" %(freq[z], lam[0], lam[1], lam[2], lam[3]), color = shades[z])
# 	Eline[q], = ax.plot(k, Efft, linewidth=2, label = r"$ s-SNOM$", color = 'black')
# 	# if (q == 0):
# 		# handles, labels = ax.get_legend_handles_labels()
# 	handles, labels = ax.get_legend_handles_labels()
# 	#plt.savefig("EFieldXSec43THz.pdf")
# 	# ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
# 	# ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
# 	ax.set_xlim(0, 5e6)
# 	ax.set_ylim(0,5)
# 	plt.setp(ax.spines.values(), linewidth=1)
# 	plt.tick_params(left = False, bottom = False)
# 	ax.set_xlabel(r"$ \rm k \ (m^{-1})$", fontsize = '35')
# 	ax.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. units)$", fontsize = '20')
# 	z = z -1

# 	lam1[q] =ax.scatter(peakk[0], peakEfft[0], s=25,edgecolors = 'black', c='r', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[0])		
# 	lam2[q] =ax.scatter(peakk[1], peakEfft[1], s=25,edgecolors = 'black', c='yellow', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[1])		
# 	lam3[q] =ax.scatter(peakk[2], peakEfft[2], s=25,edgecolors = 'black', c='lime', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[2])		
# 	lam4[q] =ax.scatter(peakk[3], peakEfft[3], s=25,edgecolors = 'black', c='cyan', zorder = 10, label = r"$%2.2f \ \mu m$" %lam[3])		


# 	q = q + 1

# # for q in range (0, 6):
# # 	handles.append(Eline[q])
# for q in range (0, 1):
# 	handles.append(lam1[q])
# 	# handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)}
# 	# temp = lam1[q]
# 	# labels.append(temp.get_label)
# for q in range (0, 1):
# 	handles.append(lam2[q])
# for q in range (0, 1):
# 	handles.append(lam3[q])
# for q in range (0, 1):
# 	handles.append(lam4[q])
# # print(handles)
# # # sort both labels and handles by labels
# # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

# ax.legend(handles = handles, ncol = 5, loc = 'center right', fancybox= True, shadow = True, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),

# ax1 = ax.twinx()


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
		# print(min(Full[i]))

	# print(max(np.sqrt(Ex)))


c0 = 3e8
threshold = 0.0
numpad = 100000
Eline = {}
lam1  = {}
lam2  = {}
lam3  = {}
lam4  = {}
MyLegend = []
# for z in range(6, 0):
z = 3
q = 0

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

while (z > 2):#(-1)):

	print(freq[z])
	E = abs(Line)
	x = np.linspace(0,  3.479e-6, Nx)

	# z = z
	# xE = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )

	padE = np.pad(E, numpad, mode='constant')
	# Efft = np.fft.fft(padE, len(padE))
	# fs = 1/(x[1]*len(padE))
	# f = np.arange(0, fs*len(padE), fs)
	# # lam = 1/f
	n = padE.size 
	dx = x[1]
	dk = 1/(n*dx)
	print(dx)
	k = np.arange(0,n*dk, dk)


	Efft = abs(np.fft.fft(padE))

	# Fourier transform data and take absolute value
	# Efft = abs(fft(pe))

	# print("Excitation Frequency", 1/(lam/1e4), "cm^-1")

	# Find peaks and store peak data
	counter = 0
	lam = []
	peakk = []
	peakEfft = []

	for i in range(2,n//2):
	    if ((Efft[i] > Efft[i-1]) & (Efft[i] > Efft[i+1]) & (Efft[i] > threshold)&(counter < 5)):
	        print(Efft[i], 1e6/k[i], "um", k[i], "cm-1")
	        peakk.append(k[i])
	        peakEfft.append(Efft[i])
	        print(counter)
	        if ((counter < 5)&(counter >= 0)):
	        	lam.append(1e6/k[i])
	        	# print(lam[counter - 1])
        	counter = counter + 1

	# ax1.plot(k, Efft, linewidth=2, label = r"$ \rm %s \ \mu m , \ \lambda_{exciton} = %2.2f \ \mu m , \  %2.2f \ \mu m , \  %2.2f \ \mu m , \  %2.2f \ \mu m  $" %(freq[z], lam[0], lam[1], lam[2], lam[3]), color = shades[z])
	Eline[q], = ax1.plot(k, Efft, linewidth=2, label = r"$ \rm FDTD $", color = 'red')
	# if (q == 0):
		# handles, labels = ax1.get_legend_handles_labels()
	handles, labels = ax1.get_legend_handles_labels()
	#plt.savefig("EFieldXSec43THz.pdf")
	# ax1.axvline(x = -0.68, linestyle = "dashed", color = 'black')
	# ax1.axvline(x =  peakk[0], linestyle = "dashed", color = 'black')
	# print(peakk[0])
	ax1.set_xlim(0, 2e7)
	ax1.set_ylim(0,1.5e-20)
	plt.setp(ax1.spines.values(), linewidth=1)
	plt.tick_params(left = False, bottom = False)
	ax1.set_xlabel(r"$ \rm k \ (m^{-1})$", fontsize = '35')
	ax1.set_ylabel(r"$\rm \vert E \vert \ Field \ (V m^{-1})$", color = 'red', fontsize = '20')
	z = z -1

	lam1[q] =ax1.scatter(peakk[0], peakEfft[0], s=25,edgecolors = 'black', c='r', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[0])		
	lam2[q] =ax1.scatter(peakk[1], peakEfft[1], s=25,edgecolors = 'black', c='yellow', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[1])		
	lam3[q] =ax1.scatter(peakk[2], peakEfft[2], s=25,edgecolors = 'black', c='lime', zorder = 10, label = r"$%2.2f \ \mu m , \ $" %lam[2])		
	lam4[q] =ax1.scatter(peakk[3], peakEfft[3], s=25,edgecolors = 'black', c='cyan', zorder = 10, label = r"$%2.2f \ \mu m$" %lam[3])		


	q = q + 1

# for q in range (0, 6):
# 	handles.append(Eline[q])
for q in range (0, 1):
	handles.append(lam1[q])
	# handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)}
	# temp = lam1[q]
	# labels.append(temp.get_label)
for q in range (0, 1):
	handles.append(lam2[q])
for q in range (0, 1):
	handles.append(lam3[q])
for q in range (0, 1):
	handles.append(lam4[q])
# print(handles)
# # sort both labels and handles by labels
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

ax1.legend(handles = handles, ncol = 5, loc = 'upper right', fancybox= True, shadow = True, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),




plt.savefig("6_8FFT2_46umPitch.pdf")
plt.savefig("6_8FFT2_46umPitch.png")
#abs(Full*(dt/T))

