#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#opening a file that will store all the values for our PBG diagram
#Sweep = np.zeros((11,2000), dtype=np.double)


fig, ax = plt.subplots(figsize=(9,5),constrained_layout=True)
# plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
ylabel("Absorption", fontsize = '35')   
xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '35')

#xlim(6,7)
#ylim(0.245, 0.275)
numbers = numbers = [2.24, 2.281, 2.29, 2.3, 2.36, 2.42, 2.48, 2.51, 2.56, 2.6]
Peak = []
for i in range (0,9):
    E1 = "%s" %numbers[i]
    E2 = "PitchRT.txt"
    E = E1 + E2
    print E
    Lambda, R,T,RT = loadtxt(E, usecols=(0,1,2,3), skiprows= 0, unpack =True)
    print len(RT)
    # Lambda = Lambda/1e6
    #plt.plot(Lambda,R, label = "R")
    #plt.plot(Lambda,T, label = "T")
    # if (i == 1):
        # plt.plot(Lambda,(1-RT), linewidth = 5, color = "blue", label = r"$\rm d_{CC} = 2.29  \ \mu m$")# %numbers[i])#(2.0+pitch/100.0))   
    if(i == 1):
        plt.plot(Lambda, (1-RT), linewidth = 5, color = "red", label = r"$d\rm_{CC} = 2.281  \ \mu m$")
        # plt.plot(Lambda, -1*T, linewidth = 5, color = "limegreen", label = r"$\rm pitch = 2.%d \ \mu m$" %numbers[i])
    else:
        plt.plot(Lambda, (1-RT), linewidth = 2, color = "black", label = r"$d\rm_{CC} = %s \ \mu m$" %numbers[i])

    ax.legend(loc='upper left', fontsize='11')
    axvline(x =7.29, color = 'black')
    # plt.tight_layout()
    Peak.insert(i, min(RT[0:300]))

# def func(d, m, c):
#     return m*d+ c
# popt, pcov = curve_fit(func, log(np.asarray(numbers)) , log(np.asarray(Peak)))


# x = np.asarray(numbers)
# a, c = polyfit(numbers, Peak, 1)
# yfit = a*x + c 
# plt.plot(x,yfit, label = "Fitted Curve", color = "black", linewidth = 3)
# plt.plot(numbers, Peak, 'ro', markersize = 15, markeredgecolor = "black", markerfacecolor = "red", label = "Absorption peak")
plt.savefig("AgPatternPitchSweepAbsJune9.pdf")
plt.savefig("AgPatternPitchSweepAbsJune9.png")


# plt.show()
#plt.clr()