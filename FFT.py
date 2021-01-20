#lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *
import matplotlib.pyplot as plt
#P = np.zeros((4,58238), dtype=np.complex64)
PSD = np.zeros((4, 57238), dtype=np.complex64)
# for z in range (1,2):
#     E1 = "T%dTime" %z
#     E = E1 + '.txt'
#     print E
    
#     #print i
#     #while (i < 11):
#         #print i
#     Time  = loadtxt(E, delimiter= ", ", usecols=(0,), skiprows=3 , unpack =True)
#     #print i
#     EField = loadtxt(E, usecols=(1,), skiprows= 3, unpack =True)
#     #E0  = np.zeros(len(E0R), dtype=np.complex64)
#     # for m in range (0, len(E0R)):
#     #     E0[m] = E0R[m] + 1j*E0I[m]
#     #P[z] = E0
#     #i = i+1
#     #print i
#     M = len(EField)
#     #print E
#     print M
#     #Mnew = M - 2000
#     #
#     for i in range (1,2):
#          E0chop = EField[1000: M]
#          Mnew = len(E0chop)
#          E0_fft = fft(E0chop, Mnew)
#          PSD[z-1] = abs(E0_fft)**2
# #     #    print M

# sumPSD  = np.zeros(Mnew)
# for j in range (0,Mnew):
#     for i in range (0, 4):
#         sumPSD[j] = sumPSD[j] + PSD[i][j]
#             #plot(sumPSD)
#         #show()

c = 3e8
Time_S = loadtxt("TimeSource2.txt", delimiter= ", ", usecols=(0,), skiprows=3 , unpack =True)
E_inc = loadtxt("TimeSource2.txt", usecols=(1,), skiprows= 3, unpack =True)

EIfft = ifft(E_inc, len(Time_S))
dtI = 75.51e-15/len(Time_S)
fIs = 1/(max(Time_S)*1e-15)
print fIs
fI = fIs*arange(0, len(Time_S))/len(Time_S)
print Time_S
LamI = c/fI
print len(LamI)
# print len(EIfft)


#print "max lamI is %g" %max(lamI)
# dt = ((max(Time) - Time[1000])/(len(Time) - 1000))*1e-15
# print max(Time)


# print dt
# fs = 1/dt

# f = fs*arange(0,Mnew)/Mnew
# lam = c/f[0: len(f)]
#plot(f, np.log(sumPSD))
#sender = (sumPSD)
#show()
fig, ax = plt.subplots(figsize=(12,9))
#ylim(0, 1)
#xlim(5,10)
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
xlabel(r"$ \rm Wavelength (\mu m)$", fontsize = '30')
ylabel(r'$\rm Power \ Spectral \ Density$', fontsize = '30')
plt.setp(ax.spines.values(), linewidth=2)
plt.plot(fI, abs(E_inc)**2, linewidth = 3) #abs(E_inc/max(E_inc))**2
#plt.plot(lam*1e6, sumPSD, linewidth = 3)
plt.savefig("SignalLumericalAg_Si.png")
plt.savefig("SignalLumericalAg_Si.pdf")
plt.show()
# ff = open(E1 + "fft.txt", 'w')

# for k in range (0, Mnew):
#         thang1 = str(f[k])
#         thang2 = str(sender[k])
#         ff.write(thang1)
#         ff.write("\t")
#         ff.write(thang2)
#         ff.write("\n")
# ff.close()




