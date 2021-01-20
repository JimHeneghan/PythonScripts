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
for i in range (3,7):
    O = i
    E1 = "MeshOrder%d" %O
    E2 = ".txt"
    E = E1 + E2
    Lambda = loadtxt(E, usecols=(0,), skiprows= 0, unpack =True)
    R = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
    tick_params(width=2, labelsize=20)
    ylabel("R", fontsize = '30')   
    xlabel("Wavelength (nm)", fontsize = '30')
    xlim(6,9)
    #ylim(0.245, 0.275)
    plt.plot(Lambda, R, label = "Mesh Order = %d" %O)
    ax.legend(loc='upper right', fontsize='10')
    plt.tight_layout()

plt.savefig("MeshOrderSweepRZoom.pdf")
plt.savefig("MeshOrderSweepRZoom.png")
    
plt.show()
#plt.clr()