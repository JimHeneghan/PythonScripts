import numpy as np
from scipy import *
from pylab import *
import math
import cmath
#from numpy import ctypeslib
#from ctypes import *

eps_inf = 4.87
s = 1.83
omega_nu = 137200 # in m
c = 3e8; 
# in m
k = linspace(120000, 170000, 2000)
gamma = 500.01

d = 60e-9

#Permittivity of hBN
eps = eps_inf + s*(omega_nu**2)/(omega_nu**2 - 1j*gamma*k - k**2)
n = np.sqrt(eps)

#n2 is the refractive index of got from (https://refractiveindex.info/?shelf=main&book=BaF2&page=Li)
# the constants are in microns and will convert k appropriately
#lam = linspace(0, 15, 2000)
lam = (1/(k))*1e6

n2 = np.zeros(2000, dtype=np.complex64) 
n2 = np.sqrt(1 + 0.33973 + (0.81070*lam**2)/(lam**2 - 0.10065**2) + (0.19652*lam**2)/(lam**2 - 29.87**2) + (4.52469*lam**2)/(lam**2 - 53.82**2))

print n2[1] 
# assuming the impedance of free space has already been cancelled out
eta1 = (1/n)
eta2 = 1/n2

#wave vector in the medium
k1 = k*n

ro1 = (eta1 - 1)/(1 + eta1)
ro2 = (eta2 - eta1)/(eta2 + eta1)

# since we're hoping that any reflected wave coming from the back of the substrate
# will be negligible we can use eqn 5.4.3 in Orfanidis
#R = np.zeros(2000, dtype=np.complex64)



R = (ro1 + ro2*np.exp(2*1j*k1*d))/(1 + ro1*ro2*np.exp(2*1j*k1*d))

print min(lam)
plot(k/100, abs(R))
show()
#plot(k/100, real(eps))
#show()
plot(lam, n2)
show()
