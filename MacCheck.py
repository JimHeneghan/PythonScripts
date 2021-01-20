import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

NFREQs = 500
Nx = 10
Ny = 10

mpl.rcParams['agg.path.chunksize'] = 10000
#all units are in m^-1
wp = 1.15136316e16
gamma = 9.79125662e13
d = 50e-9
imp0 = 376.730313
w = linspace (100e12, 3600e12,  100)
# k0 = k*1e12
c0 = 3e8
eps1 = 1 + (wp*wp)/(w*(1j*gamma-w))
dx = 5e-8
dy = 5e-8
n1 = np.sqrt(eps1)

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
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

# B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
# C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
# Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))

ExR, ExI, EyR, EyI, EzR, EzI = np.loadtxt("PlaneRef.txt",  usecols=(5,6,7,8,9,10), skiprows= 1, unpack =True)
freq= np.loadtxt("PlaneInc.txt",  usecols=(0), skiprows= 1, unpack =True)
freq = freq*1e12

EyRef = EyR + 1j*EyI

EyRef = np.reshape(EyRef, (NFREQs, Nx, Ny), order='C')

EyThing = np.zeros(NFREQs, dtype = np.double)
for t in range(0, NFREQs):
	EyThing[t] = abs(EyRef[t, 5, 5])


fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0,10)
# plt.ylim(0,1)


plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='yellow', linewidth = 4)

plt.plot((c0/freq)*1e6, EyThing)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
plt.savefig("XfNorm2.pdf")
plt.savefig("XfNorm2.png")
