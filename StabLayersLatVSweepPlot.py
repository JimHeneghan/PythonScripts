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
tick_params(width=2, labelsize=20)
ylabel("T", fontsize = '30')   
xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
#xlim(6,7)
#ylim(0.245, 0.275)

for i in range (1,12):   
    pitch = (30.0+((i-1)/5.0)*10.0)
    E1 = "Pitch2_%d" %pitch
    E2 = "umStabLayers.txt"
    E = E1 + E2
    print E
    Lambda, R,T,RT = loadtxt(E, usecols=(0,1,2,3), skiprows= 0, unpack =True)
    Lambda = Lambda/1e6
    #plt.plot(Lambda,R, label = "R")
    #plt.plot(Lambda,T, label = "T")
    plt.plot(Lambda,T, label = r"$\rm pitch = %1.2f \mu m$" %(2.0+pitch/100.0))   
    ax.legend(loc='upper right', fontsize='15')
    axvline(x =7.29, color = 'black')
    plt.tight_layout()
#plt.savefig("StabLAyersPitchSweep.pdf")
#plt.savefig("StabLAyersPitchSweep.png")
plt.show()
#plt.clr()