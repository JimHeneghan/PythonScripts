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
ylabel("T", fontsize = '30')   
xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
ax.legend(loc='upper right', fontsize='10')
#plt.tight_layout()
Lambda, R, T, RT = loadtxt("26nmdxRT.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)
#R = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
xlim(5,9)
ylim(0, 1)

plt.plot(Lambda,R, label = "R", color = "red", linewidth = 3)
plt.plot(Lambda,T, label = "T", color = "black", linewidth = 3)
plt.plot(Lambda,RT, label = "R+T", color = "limegreen", linewidth = 3)
plt.setp(ax.spines.values(), linewidth=2)  
ax.legend(loc='upper left', fontsize='30')
ax.axvline(x = 7.295, color = 'black', linewidth = 2)
ax.axvline(x = 7.58, color = 'black', linewidth = 2)
plt.savefig("Weird26nmdxRT.pdf")
plt.savefig("Weird26nmdxRT.png")
    
#plt.show()
#plt.clr()
