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
for i in range (1,9):
    layers = (32*(i))
    E1 = "%dStabalized" %layers
    E2 = "Layers.txt"
    E = E1 + E2
    Lambda = loadtxt(E, usecols=(0,), skiprows= 0, unpack =True)
    R = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
    #xlim(6,7)
    #ylim(0.245, 0.275)
    
    if (i ==1 or i ==8):
        plt.plot(Lambda, R, label = "Number of Stabilized Layers = %d" %layers)
        
ax.legend(loc='upper right', fontsize='30')
plt.savefig("FirstLastStabilizedLayers.pdf")
plt.savefig("FirstLastStabilizedLayers.png")
    
plt.show()
#plt.clr()