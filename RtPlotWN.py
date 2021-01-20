#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#opening a file that will store all the values for our PBG diagram
#Sweep = np.zeros((11,2000), dtype=np.double)
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
# ylabel("T", fontsize = '30')   
ax.set_xlabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '30')
ax.legend(loc='upper right', fontsize='10')
plt.tight_layout()
Lambda, R, T, RT = loadtxt("hBNNewGamma.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)
#R = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
plt.xlim(1200,1700)
plt.ylim(0,1)
# plt.ylim(0,1)
print len(RT)
print min(RT[0:1000])
wn = (1/(Lambda*1e-6))/100
plt.plot(wn,R, label = "R", color = "red", linewidth = 3)
plt.plot(wn,T, label = "T", color = "black", linewidth = 3)
plt.plot(wn,RT, label = "R+T", color = "limegreen", linewidth = 3)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='30')
ax.axvline(x = ((1/(7.295*1e-6)))/100, color = 'black', linewidth = 2)
plt.savefig("hBNonlyumRTLineWN.pdf")
plt.savefig("hBNonlyumRTLineWN.png")
    
plt.show()
#plt.clr()
