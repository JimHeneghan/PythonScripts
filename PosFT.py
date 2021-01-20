import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, axs = plt.subplots(2, 3, figsize = (16, 8),constrained_layout=True)
fig, ax1 = plt.subplots(1, figsize = (16, 8),constrained_layout=True)

z = 43
rain = ['red', 'orange', 'yellow', 'limegreen', 'cyan']
for ax in axs.flat:

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

	dx = 10e-9

	print(len(Full[200][15:215]))
	# fs = 1/(dx*len(Full[200][15:215]))
	
	f = dx*np.arange(0,len(Full[200][15:215]))
	print(max(f))
	print(len(f))
	Xfft = np.fft.fft(Full[200][15:215], len(Full[200][15:215]))
	lam = (3e8/(z*1e12))*1e6
	wn = (z*1e12/3e8)/100
	ax.set_title(r"$ \rm  %.2f \ (\mu m), \ %.2f \ (cm^{-1}), \ %d \ (THz)$" %(lam, wn, z),  fontsize = '10')
	ax.set_xlabel(r"$ \rm \lambda \ (n m)$", fontsize = '20')
	ax.set_ylabel(r"$ \rm Intensity $", fontsize = '20')
	ax.tick_params(direction = 'in', width=2, labelsize=20)
	plt.setp(ax.spines.values(), linewidth=2)
	# ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
	# ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
	# ax.set_xlim(0, 800)
	# ax.set_ylim(0, 25)
	# ax.plot(f*1e9, abs(Xfft), color = 'black', linewidth = 4)

	for i in range (1,199):
		if ((Xfft[i-1] < Xfft[i])and(Xfft[i] > Xfft[i+1])):
			peak.append(f[i]*1e9)


	ax1.plot(np.repeat(z, len(peak)), peak, '.', color = 'black', linewidth = 4)

	# plt.show()
	# plt.clf()
	# print(len(x))
	# lam = []
	# hz = []
	# for i in range (1, len(x)):
	# 	lam.append(x[i]*1.0 - x[i-1]*1.0)
	# 	print(lam[i-1])
	# 	hz.append((3e8/(z*1e12))*1e6)
	z = z + 1
	# plt.plot(hz, lam, "o")
plt.savefig("PosFT/Band.png")	

# plt.show()