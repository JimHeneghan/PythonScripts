import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
rain = ['red', 'orange', 'yellow', 'limegreen', 'cyan']
fig, ax = plt.subplots(1, figsize = (6.4, 5.74),constrained_layout=True)
for z in range(44, 45):
	Field = "%dHz.txt" %z
	print(Field)
	z = z
	Full = np.zeros((401, 231), dtype = np.double)

	X = np.linspace(0,  2.31, 231)
	Y = np.linspace(0,  4.01, 401)
	for i in range (0, 400):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 639, unpack =True )
		Full[i] = E 
	peak = []
	x = []
	xint = []
	q = 1
	for i in range (47,183):
		if (((Full[200][i-1]**2) < (Full[200][i])**2)and((Full[200][i]**2) > (Full[200][i+1])**2)):		
			peak.append(Full[200][i]**2)
			xint.append(i)
			# print(xint[q-1])
			x.append((i/100.0) - 1.15)
			q = q + 1
	lam = (3e8/(z*1e12))*1e6
	wn = (z*1e12/3e8)/100
	ax.set_title(r"$ \rm  %.2f \ (\mu m), \ %.2f \ (cm^{-1}) \ %d \ (THz)$" %(lam, wn, z),  fontsize = '20')
	ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax.set_ylabel(r"$ \rm Intensity $", fontsize = '20')
	ax.tick_params(direction = 'in', width=2, labelsize=20)
	plt.setp(ax.spines.values(), linewidth=2)
	ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
	ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
	ax.plot(X[15:215] - 1.15, Full[200][15:215]**2, color = 'black', linewidth = 4)
	for i in range (1, len(xint)):
		ax.plot(X[xint[i-1]:xint[i]] - 1.15, Full[200][xint[i-1]:xint[i]]**2, color = rain[i-1], label = r"$\rm \lambda = %.0f \ (nm)$" %((x[i] - x[i-1])*1000))
		print(xint[i-1])
	ax.plot(x, peak, "o", color = 'red')
	ax.legend(loc='upper center', fontsize='15')
	
plt.savefig("Peaks/44HzPeaksColors.png")