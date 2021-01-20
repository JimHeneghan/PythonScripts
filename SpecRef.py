import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
# import matplotlib as mpl

######################################################################################
################################## Tm Calculation ####################################
######################################################################################

w = linspace (100e12, 3600e12,  100)
eps_inf = 1.0
eps_s   = 343.744
tau     = 7.90279e-15
sigma   = 3.32069e6
eps0 = 8.85418782e-12

eps1 = eps_inf + (eps_s-eps_inf)/(1+1j*w*tau) + sigma/(1j*w*eps0)

n1 = np.sqrt(eps1)

d = 50e-9
imp0 = 376.730313
# k0 = k*1e12
c0 = 3e8
dx = 50e-9
dy = 50e-9
nref = 1.0
ntra = math.sqrt(3.8)
#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*(w/c0)#*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0*ntra
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

# B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
# C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
# Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))


######################################################################################
########### Reflection Spectrum Calculated from Time Domain pt Sources ###############
######################################################################################
numT = 50000
numE = 50000
Ix  = np.zeros((numT, numE), dtype = np.double)
Rx= np.loadtxt("Ref10.txt", usecols=(1), skiprows= 0, unpack =True)
# R0 = np.loadtxt("../Vac/Ref/Ref.txt", usecols=(0,), skiprows= 1, unpack =True )
Ix[0], Ix[1] = np.loadtxt("Inc10.txt", usecols=(0,1), skiprows= 0, unpack =True )
# Tx = np.loadtxt("Trans.txt", usecols=(2), skiprows= 1, unpack =True )

# mpl.rcParams['agg.path.chunksize'] = 10000

# plt.plot(Rx)
# # plt.ylim(0,1e-5)
# plt.savefig("Rx.png")
# plt.clf()


# plt.plot(Ix)
# plt.savefig("Ix.png")
# plt.clf()


# # print(len(Ry))
# c0 = 3e8
# ddx = 5e-9
# dt = 1.651388e-16 #ddx/(2*c0)
print(Ix[1])
print(Ix[0])
print(len(Ix[1]))
# Ix = np.asarray(Ix)
print(Ix[1])
print(Ix[0])


# Ix = np.reshape(Ix, (50000, 50000), order='C')
print(Ix)
print("\n \n \n")

freq  = np.linspace(30e12, 300e12, int(len(Ix[1])/2))  #1/Ix[0] #fft(Ix[0])
Ixfft = fft(Ix[1])
Rxfft = fft(Rx)
print(freq)
print(Ixfft)
# print(len(Ixfft))
# # Ixfft = Ixfft.T
# print(len(Ixfft))
# print(Ixfft[1])
# print(Ixfft[0])
# # Ixfft = abs(fft(Ix))

# Rxfft = np.fft.fft(Rx)


# Txfft = np.fft.fft(Tx, len(Rx))


# fs = 1/(dt*len(Rx))
# f = fs*np.arange(0,len(Rx))

# print(f)

# lam = c0/(f)

# plt.xlim(0, 10)
# plt.ylim(0,1)
# plt.plot((c0*1e6)/freq, abs(Ixfft[0:int(len(Ixfft)/2)]))
# # plt.savefig("IFFt_NoTrans.png")
# plt.show()
# plt.clf()

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("T", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0, 10)
plt.ylim(0,1)
plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='yellow', linewidth = 4)
plt.plot((c0*1e6)/freq[0:-5000], ((abs(Rxfft[0:(len(Rx)/2)-5000]))**2)/(abs(Ixfft[0:((len(Rx)/2)-5000)]))**2, label = "R", color = "red", linewidth = 3)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='30')
# ax.axvline(x = 7.295, color = 'black', linewidth = 2)
plt.savefig("XFAgOnSiO2_2.png")
# plt.show()
plt.clf()
