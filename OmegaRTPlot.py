#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#Script to Plot the Coupled RT Plots in terms of frequency for
#83 nm hBN in Vacuum
#Uncoupled Hex Patterned Ag with Pitch = 2.3, 2.51 
#83 nm hBN on Hex Patterned Ag with Pitch = 2.3, 2.51 	
fig, axs = plt.subplots(3, 2, figsize = (24, 14),constrained_layout=True)
rc('axes', linewidth=2)
q = 0
n = 0

for z in range (1, 7):
	Dat = "%d.txt" %z   

	plt.setp(axs[q, n].spines.values(), linewidth=2)
	axs[q, n].tick_params(direction = 'in', width=2, labelsize=20)
	axs[q, 0].set_ylabel("T", fontsize = '30')   
	axs[2, n].set_xlabel(r"$\rm Frequency \ (Hz)$", fontsize = '30')
	Lambda, R, T, RT = loadtxt(Dat, usecols=(0,1,2,3), skiprows= 0, unpack =True)

	if (z == 3 or z == 4):
		Lambda = Lambda*1e-6 
		T = -1*T
		RT = R + T

	omega = 3e8/(Lambda*1e-6)
	lim1 = 3e8/(9*1e-6)
	lim2 = 3e8/(5*1e-6)
	axs[q,n].set_xlim(lim1, lim2)
	axs[q,n].set_ylim(0,1)
	if (q == 2):
		axs[q,n].set_ylim(0.87,1.028)
	axs[q, n].plot(omega,R, label = "R", color = "red", linewidth = 3)
	axs[q, n].plot(omega,T, label = "T", color = "black", linewidth = 3)
	axs[q, n].plot(omega,RT, label = "R+T", color = "limegreen", linewidth = 3) 
	const = 3e8/1e-6
	axs[0, 1].legend(loc='center right', fontsize='20')
	axs[q, n].axvline(x = const/7.295, color = 'black', linewidth = 2, label = "hBN Resonance")
	axs[q, n].axvline(x = const/6.767, color = 'blue', linewidth = 2, label = "Uncoupled\nExperimental\nResonance")
	if (n == 0):
		axs[q, n].axvline(x = const/7.59, color = 'fuchsia', linewidth = 2, label = "Coupled\nResonance")
	else:
		axs[q, n].axvline(x = const/8.145, color = 'fuchsia', linewidth = 2, label = "Coupled\nResonance")

	if (n == 0): 
		n = + 1
	else:
		n = 0
		q = q + 1

plt.savefig("FreqPlotCoupledRTMar6Zoom.pdf")
plt.savefig("FreqPlotCoupledRTMar6Zoom.png")
	    
plt.show()

