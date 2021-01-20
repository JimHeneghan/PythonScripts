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
for i in range (1,11):
    time = 5*i
    E1 = "Time%g" %time
    E2 = "ps.txt"
    E = E1 + E2
    Lambda = loadtxt(E, usecols=(0,), skiprows= 0, unpack =True)
    T = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
    tick_params(width=2, labelsize=20)
    ylabel("R", fontsize = '30')   
    xlabel("Wavelength (nm)", fontsize = '30')
    xlim(6,7)
    ylim(0.2, 0.3)
    plt.plot(Lambda, T, label = "Run Time = %g ps" %time)
    ax.legend(loc='upper right', fontsize='10')
    plt.tight_layout()

plt.savefig("TimeSweepTZoom.pdf")
plt.savefig("TimeSweepTZoom.png")
#plt.show()
