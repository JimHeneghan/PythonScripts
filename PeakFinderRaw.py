import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
rain = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
wl = ['4.7', '5.5', '5.7', '5.9', '6.1', '6.3', '6.4', '6.5', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']

for z in range(0, len(wl)):
	fig, ax = plt.subplots(1, figsize = (15, 9),constrained_layout=True)

	Field = "%s_um.txt" %wl[z]
	Fname = "%s_um" %wl[z]
	print(Field)
	print(Fname)

	z = z
	# Full = np.zeros((401, 231), dtype = np.double)

	# for i in range (0, 400):
	# 	E = np.loadtxt(Field, usecols=(i,), skiprows= 639, unpack =True )
	# 	Full[i] = E 

	X, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )
	peakx = []
	peakE = []

	for i in range(0,len(E)-1):
	    if ((E[i]**2 > E[i-1]**2) & (E[i]**2 > E[i+1]**2)):
	        # print(Efft[i], 1.0/k[i], "um", 2*math.pi*k[i]*1e4, "cm-1")
	        peakx.append(X[i]*1e6)
	        peakE.append(E[i]**2)
	
	avg = 0.0
	for i in range (0, len(peakx) - 1):
		avg = avg + 2*(peakx[i+1] - peakx[i])

	avg = avg/(len(peakx) - 1)
	# x = []
	# peak = []
	# xint = []
	# q = 1
	# for i in range (0,len(E) -1):
	# 	if (((E[i-1]**2) < (E[i])**2)and((E[i]**2) > (E[i+1])**2)):		
	# 		peak.append(E[i]**2)
	# 		xint.append(i)
	# 		# print(xint[q-1])
	# 		x.append((i/100.0) - 1.15)
	# 		q = q + 1
	# lam = (3e8/(z*1e12))*1e6
	# wn = (z*1e12/3e8)/100
	ax.set_title(r"$%s \ (\mu m)$" %wl[z],  fontsize = '20')
	ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax.set_ylabel(r"$ \rm Intensity $", fontsize = '20')
	ax.tick_params(direction = 'in', width=2, labelsize=20)
	plt.setp(ax.spines.values(), linewidth=2)

	# ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
	# ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
	ax.plot(X*1e6, E**2, color = 'black', linewidth = 4)
	ax.scatter(peakx, peakE, s=25,edgecolors = 'black', c='r', zorder = 10, label = r"$ Average \ \lambda = %2.2f \ \mu m$" %avg )		

	# print(len(xint))
	# for i in range (1, len(xint)):
	# 	print(i)
	# 	ax.plot(X[xint[i-1]:xint[i]], E[xint[i-1]:xint[i]]**2, color = rain[i-1])#, label = r"$\rm \lambda = %.0f \ (nm)$" %((x[i] - x[i-1])*1000))
	# 	# print(xint[i-1])
	# ax.plot(x, peak, "o", color = 'red')
	ax.legend(loc='upper right', fontsize='15')
	
	plt.savefig("EyePeak/"+ Fname + "Peaks.png") 
	plt.clf()