import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

for z in range(43, 51):
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