from numpy import *
from scipy import *
from pylab import *
import matplotlib.pyplot as plt
import math
for z in range (1, 50):
    R1 = "SilverSiHexRadFullSweep%d" %z
    Ref = R1 + '.txt'
    freq = loadtxt(Ref, usecols=(0,),  unpack =True)
    R = loadtxt(Ref, usecols=(1,),  unpack =True)
    plt.figure()
    axvline(x =6.9, color = 'black')
    q = 0.5 + z*9.8e-3
    Tit = "Hole Radius %g $\mu$m" %q
    plt.title(Tit)
    plt.ylabel("Reflectance")
    plt.xlabel("Wavelength ($\mu$m)") 
    plt.xlim(5.5, 7.0)
    plt.plot(3e8/freq*1e6, R)
#    show()
    R2 = "SilverSiHexRadSweep%d" %z
    plt.savefig(R1 + '.pdf')
