import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *
import matplotlib.pyplot as plt
Time  = loadtxt("T2Time.txt", delimiter= ", ", usecols=(0,), skiprows=3 , unpack =True)
EField = loadtxt("T2Time.txt", usecols=(1,), skiprows= 3, unpack =True)

fig, ax = plt.subplots(figsize=(12,9))
#plt.tight_layout()
#plt.setp(ax.spines.values(), linewidth=2)
#ylim(0, 0.005)
#xlim(5,10)
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
xlabel(r"$ \rm Time (ps)$", fontsize = '30')
ylabel(r'$\rm E \ Field$', fontsize = '30')
plt.setp(ax.spines.values(), linewidth=2)
plt.plot(Time, EField, linewidth = 3)
plt.savefig("TimeDataFromLumericalAg_Si.png")
plt.savefig("TimeDataLumericalAg_Si.pdf")
plt.show()
