#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#opening a file that will store all the values for our PBG diagram
#Sweep = np.zeros((11,2000), dtype=np.double)


fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
ylabel("R+T", fontsize = '35')   
xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '35')

#xlim(6,7)
#ylim(0.245, 0.275)
numbers = numbers = [18, 26, 34, 42, 50]
for i in range (0,5):
    E1 = "%s" %numbers[i]
    E2 = "nmdxRT.txt"
    E = E1 + E2
    print E
    Lambda, R,T,RT = loadtxt(E, usecols=(0,1,2,3), skiprows= 0, unpack =True)
    # Lambda = Lambda/1e6
    #plt.plot(Lambda,R, label = "R")
    #plt.plot(Lambda,T, label = "T")
    
    if (i == 0):
        plt.plot(Lambda, RT, linewidth = 5, color = "red", label = r"$\rm dx=dy= %s nm$" %numbers[i])#(2.0+pitch/100.0))   
    elif(i == 1):
        plt.plot(Lambda, RT, linewidth = 2, color = "orange", label = r"$\rm dx=dy=%s nm$" %numbers[i])
    elif(i == 2):
        plt.plot(Lambda, RT, linewidth = 2, color = "green", label = r"$\rm dx=dy= %s nm$" %numbers[i])
    elif(i == 3):
        plt.plot(Lambda, RT, linewidth = 2, color = "blue", label = r"$\rm dx=dy= %s nm$" %numbers[i])
    else:
        plt.plot(Lambda, RT, linewidth = 2, color = "purple", label = r"$\rm dx=dy= %s nm$" %numbers[i])
   

# Lambda, R,T,RT = loadtxt("hBN_Ag_Si2.31umRT.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)     

# plt.plot(Lambda, RT, linewidth = 5, color = "limegreen", label = r"$\rm dx=dy=10dz= 1 nm$")
# plt.plot(Lambda, RT, linewidth = 1, color = "black")#, label = r"$\rm dx=dy=10dz= 1 nm$")

plt.ylim(0.5,1)
ax.legend(loc='lower left', fontsize='20')
axvline(x =7.29, color = 'black')
plt.tight_layout()
plt.savefig("AgSidxdyMeshSweepAbs.pdf")
plt.savefig("AgSidxdyMeshSweepAbs.png")
# plt.show()
#plt.clr()