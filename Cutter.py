import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(1, figsize = (4,4),constrained_layout=True)



rain = ['red', 'orange', 'yellow', 'limegreen', 'cyan']
Source = ['6.1', '6.3', '6.4', '6.5', '6.75', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0']



for z in range (0, len(Source)):
	Field = Source[z] + "Ex.txt" 
	print(Field)

	Full = np.zeros((221, 129), dtype = np.double)
	X = np.linspace(0,  2.281, 129)
	Y = np.linspace(0,  4.01, 221)

	for i in range (0, 221):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 357, unpack =True )
		Full[i] = E 
	peak = []
	x = []
	xint = []
	q = 1

	ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '15')
	ax.set_ylabel(r"$ \rm |E| \ (V m^{-1})$", fontsize = '15')
	plt.setp(ax.spines.values(), linewidth=2)
	ax.tick_params(left = False, bottom = False)
	ax.plot(X, Full[110])
	file1 = open("%sExCrossCut.txt" %Source[z], "w")
	file1.write("x (100 um) \t E (V/m) \n")
	for i in range (0, len(Full[110])):
		file1.write("%f \t %f \n" %(i*1.8, Full[110][i]))

	plt.savefig("%sEx.png" %Source[z])	

# plt.show()