#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#opening a file that will store all the values for our PBG diagram
#Sweep = np.zeros((11,2000), dtype=np.double)
fig, ax = plt.subplots(figsize=(15,9))
plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
ylabel("T", fontsize = '30')   
xlabel("Wavelength (nm)", fontsize = '30')
ax.legend(loc='upper right', fontsize='10')
plt.tight_layout()
Lambda, R, T, RT = loadtxt("AgonSi2_3PitchStabL.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)
#R = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
#xlim(6,7)
#ylim(0.245, 0.275)

plt.plot(Lambda,R, label = "R")
plt.plot(Lambda,T, label = "T")
plt.plot(Lambda,RT, label = "RT")   
ax.legend(loc='upper right', fontsize='30')
plt.savefig("StabilizedLayersExtendLong.pdf")
plt.savefig("StabilizedLayersExtendLong.png")
    
plt.show()
#plt.clr()
