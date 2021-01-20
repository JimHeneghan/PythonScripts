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
ylabel("T", fontsize = '35')   
xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '35')

xlim(5,9)
ylim(0, 1)
numbers = [2.3, 2.36, 2.42, 2.48, 2.51, 2.56, 2.6]
for i in range (0,7):   
    # pitch = (30.0+ 2*i)
    E1 = "%s" %numbers[i]
    E2 = "PitchRT.txt"
    E = E1 + E2
    print E
    Lambda, R, T, R_T = loadtxt(E, usecols=(0,1,2,3), skiprows= 0, unpack =True)
    RT = R + T
    # Lambda = Lambda/1e6
    #plt.plot(Lambda,R, label = "R")
    #plt.plot(Lambda,T, label = "T")
    # if (i == 4):
    plt.plot(Lambda, T, linewidth = 5, color = "black", label = r"$\rm T$")#(2.0+pitch/100.0)) 
    plt.plot(Lambda, R, linewidth = 5, color = "red", label = r"$\rm R$" )
    plt.plot(Lambda, RT, linewidth = 5, color = "limegreen", label = r"$\rm R \ + \ T$" )  
    # elif(i == 4):
    #     plt.plot(Lambda, -1*T, linewidth = 5, color = "limegreen", label = r"$\rm pitch = 2.%d \ \mu m$" %numbers[i])
    #     plt.plot(Lambda, R, linewidth = 5, color = "lime", label = r"$\rm pitch = 2.%d \ \mu m$" %numbers[i])
    #     plt.plot(Lambda, RT, linewidth = 5, color = "darkgreen", label = r"$\rm pitch = 2.%d \ \mu m$" %numbers[i])
    #else:
    #    plt.plot(Lambda, -1*T, linewidth = 2, color = "black", label = r"$\rm pitch = 2.%d \ \mu m$" %numbers[i])

    ax.legend(loc='center right', fontsize='30')
    ax.axvline(x = 7.295, color = 'black', linewidth = 2)
    ax.axvline(x = 6.77, color = 'black', linewidth = 2)
    #axvline(x =7.29, color = 'black')
    plt.tight_layout()
plt.savefig("AgSiPitchSweepRT1.8dz.pdf")
plt.savefig("AgSiPitchSweepRT1.8dz.png")
# plt.show()
#plt.clr()