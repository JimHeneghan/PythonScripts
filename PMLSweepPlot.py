#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#opening a file that will store all the values for our PBG diagram
#Sweep = np.zeros((11,2000), dtype=np.double)

for i in range (1,11):
    fig, ax = plt.subplots(figsize=(15,9))
    plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
    plt.setp(ax.spines.values(), linewidth=2)
    tick_params(width=2, labelsize=20)
    ylabel("R", fontsize = '30')   
    xlabel("Wavelength (nm)", fontsize = '30')
    #xlim(6,7)
    #ylim(0.245, 0.275)
    plt.tight_layout()
    
    layers = i
    E1 = "CustomLayersAlpha%d" %layers
    E2 = ".txt"
    E = E1 + E2
    Lambda, R,T,RT = loadtxt(E, usecols=(0,1,2,3), skiprows= 0, unpack =True)
    alpha = 1.0-((i-1.0)/10.0)
    Lambda = Lambda/1e6
    plt.plot(Lambda,R, label = "R")
    plt.plot(Lambda,T, label = "T")
    plt.plot(Lambda,RT, label = "RT" "\n" r"$\rm \alpha = %1.2f$" %alpha)   
    ax.legend(loc='upper right', fontsize='30')
    plt.savefig("CustomLayersAlpha%1.2f.pdf" %alpha)
    plt.savefig("CustomLayersAlpha%1.2f.png" %alpha)
    #plt.savefig("PMLSweepRZoom.pdf")
    #plt.savefig("PMLSweepRZoom.png")
    
    #plt.show()
#plt.clr()