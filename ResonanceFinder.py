#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#opening a file that will store all the values for our PBG diagram
f = open("pbg3.txt", "w")
val = 0
Sweep = np.zeros((21,1000), dtype=np.double)
for i in range (0,21):
    pitch = 2.2 + i*0.01
    E1 = "Pitch=%g" %pitch
    E2 = "um.txt"
    E = E1 + E2
    Lambda = loadtxt(E, usecols=(0,), skiprows= 1, unpack =True)
    T = loadtxt(E, usecols=(1,), skiprows= 1, unpack =True)

    TransPeak = max(T)
    Max_index = np.where(T == TransPeak)
    Sweep[i] = T
    LamMaxT = Lambda[Max_index]
    print double(LamMaxT)
#finding peaks for every point in our range
    thang1 = str(pitch)
    thang2 = str(double(LamMaxT))
    f.write(thang1)
    f.write("\t")
    f.write(thang2)
    f.write("\n")
    

f.close()
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

for i in range (0,21):
    pitch = 2.2 + i*0.01
    plot(Lambda, Sweep[i], label = r" %g $\rm \mu m$" %pitch )
pfont = {'fontname' : 'Times'} 
ylabel('Transmission', pfont, fontsize = '20')
xlabel(r'$\rm \lambda \ (\mu m)$', pfont, fontsize = '20')
ylim(0,1)
plt.tight_layout()
legend(loc= 'upper right', fontsize='15', ncol = 3)

plt.show()
