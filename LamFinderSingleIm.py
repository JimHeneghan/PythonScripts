import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax0 = plt.subplots(figsize = (3.5, 3.5),constrained_layout=True)
rain = ['red', 'orange', 'yellow', 'limegreen', 'cyan']
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
	q = 1
	for i in range (47,183):
		if ((Full[200][i-1] < Full[200][i])and(Full[200][i] > Full[200][i+1])):		
			peak.append(Full[200][i])
			x.append(i)
			q = q + 1
	

	ax0.set_aspect(401/231)
	norm = mpl.colors.Normalize(vmin=0, vmax=10)
	im = ax0.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)
	
	# cbar = fig.colorbar(im, ax=ax0)
	# cbar.set_label(label = r"$\rm E \ \ Field \ (V m^{-1})$", size = '20')
	for i in range (1, len(xint)):
		ax0.plot(X[xint[i-1]:xint[i]], ycon[xint[i-1]:xint[i]], linewidth = 4, color = rain[i-1])
	

	plt.plot(X[15:215]*100, Full[200][15:215], color = 'black')
	plt.plot(x, peak, "o", color = 'red')
	plt.savefig("Peaks/%dPeaks.png" %z)
	plt.clf()
	print(len(x))
	lam = []
	hz = []
	for i in range (1, len(x)):
		lam.append(x[i]*1.0 - x[i-1]*1.0)
		print(lam[i-1])
		hz.append((3e8/(z*1e12))*1e6)
	# plt.plot(hz, lam, "o")
	

# plt.show()