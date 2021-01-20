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

Lambda = loadtxt("256StabalizedLayers.txt", usecols=(0,), skiprows= 0, unpack =True)
R = loadtxt("256StabalizedLayers.txt", usecols=(1,), skiprows= 0, unpack =True)
xlim(5,10)
ylim(0, 1)
    
plt.plot(Lambda, R, label = "Number of Stabilized layers = 256")

LambdaPML = loadtxt("PML64layers.txt", usecols=(0,), skiprows= 0, unpack =True)       
RPML = loadtxt("PML64layers.txt", usecols=(1,), skiprows= 0, unpack =True)
plt.plot(LambdaPML, RPML, label = "Number of SA layers = 64")
ax.legend(loc='upper right', fontsize='20')    
plt.savefig("StabilizedVPML.pdf")
plt.savefig("StabilizedVPML.png")
plt.show()
#plt.clr()