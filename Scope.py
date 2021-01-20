#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *

t = loadtxt("Baby0.txt", usecols=(0,), skiprows= 1, unpack =True)
E0R = loadtxt("Baby0.txt", usecols=(1,), skiprows= 1, unpack =True)
#((1.995e-5)/(3e8*2))*


fig, ax = plt.subplots(figsize=(15,9))
plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
ylabel("E", fontsize = '30')   
xlabel("Time", fontsize = '30')
#xlim(0,10000)
#ylim(-E0R[10000],E0R[10000])


plt.plot(t, E0R)
plt.show()

#plt.savefig("MagneticPLasmaBadGain1.pdf")
#plt.savefig("MagneticPLasmaBadGain1.png")