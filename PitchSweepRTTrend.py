#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#opening a file that will store all the values for our PBG diagram
#Sweep = np.zeros((11,2000), dtype=np.double)


fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
ylabel(r"$\rm (Absorption \ Peak)^{-1} \ (\mu m)^-{1}$", fontsize = '30')   
xlabel(r"$\rm Pitch Length \ (\mu m)$", fontsize = '30')

#xlim(6,7)
#ylim(0.245, 0.275)
numbers = numbers = [2.24, 2.29, 2.36, 2.42, 2.48, 2.51, 2.56, 2.6]
Peak = []
LamFit = []
for i in range (0,8):
    E1 = "%s" %numbers[i]
    E2 = "PitchRT.txt"
    E = E1 + E2
    print E
    Lambda, R,T,RT = loadtxt(E, usecols=(0,1,2,3), skiprows= 0, unpack =True)
    print len(RT)
    # Lambda = Lambda/1e6
    #plt.plot(Lambda,R, label = "R")
    #plt.plot(Lambda,T, label = "T")
    # if (i == 0):
    #     plt.plot(Lambda, RT, linewidth = 5, color = "blue", label = r"$\rm pitch = 2.24  \ \mu m$")# %numbers[i])#(2.0+pitch/100.0))   
    # elif(i == 1):
    #     plt.plot(Lambda, RT, linewidth = 5, color = "red", label = r"$\rm pitch = 2.3  \ \mu m$")
    #     # plt.plot(Lambda, -1*T, linewidth = 5, color = "limegreen", label = r"$\rm pitch = 2.%d \ \mu m$" %numbers[i])
    # else:
    #     plt.plot(Lambda, RT, linewidth = 2, color = "black", label = r"$\rm pitch = %s \ \mu m$" %numbers[i])

    
    # axvline(x =7.29, color = 'black')
    # plt.tight_layout()
    Peak.insert(i, min(RT[0:300]))
    q = np.argmin(RT[0:300])
    print q
    LamFit.insert(i, 1/Lambda[q])
# def func(d, m, c):
#     return m*d+ c
# popt, pcov = curve_fit(func, log(np.asarray(numbers)) , log(np.asarray(Peak)))
axvline(x =2.281860295568523, color = 'black')
axhline(y = 1/7.29, color = 'black')

x = np.asarray(numbers)
# y = np.array()
a, c = polyfit(numbers, LamFit, 1)
yfit = a*x + c 
print ((((1/7.29) - c))/a)
plt.plot(x,yfit, label = "Fitted Curve", color = "black", linewidth = 3)
plt.plot(numbers, LamFit, 'ro', markersize = 15, markeredgecolor = "black", markerfacecolor = "red", label = "Simulation Absorption Peak")
plt.scatter(2.281860295568523, 1/7.29, s = 100, facecolors = "none", edgecolor = "limegreen",linewidth=3.0, zorder = 6, label = r"$\rm Predicted Resonance \ Peak \ Match \ = %5.3f  \ (\mu m) $" %2.281860295568523 )

ax.legend(loc='upper right', fontsize='20')
plt.savefig("AgPatternResTrendLam.pdf")
plt.savefig("AgPatternResTrendLam.png")


# plt.show()
#plt.clr()