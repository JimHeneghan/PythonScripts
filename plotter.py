#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *
import matplotlib.pyplot as plt

for i in range (600, 1000):	

	E1 = "Ex.%d" %i
	E = E1 + ".dat"
	Ex = loadtxt(E, usecols=(0,), skiprows= 0, unpack =True)

	x = np.linspace(0, 4999, 4999)

	fig, ax = plt.subplots(figsize=(15,9))
	plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
	plt.setp(ax.spines.values(), linewidth=2)
	tick_params(width=2, labelsize=20)
	ylabel("Ex", fontsize = '30')   
	xlabel("X", fontsize = '30')

	plt.plot(x, Ex)
	plt.savefig(E1 + ".png")
	#plt.clear()