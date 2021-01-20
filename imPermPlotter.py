#hi from the outside
from numpy import *
from scipy import *
from pylab import *
import math

lam = loadtxt("hBNgetindexConstZ.txt", usecols=(0,),  unpack =True)
n_r = loadtxt("hBNgetindexConstZ.txt", usecols=(1,),  unpack =True)
n_i = loadtxt("hBNgetindexConstZ.txt", usecols=(2,),  unpack =True)

nc = n_r + 1j*n_i
eps = nc**2

xlim(5,10)
plot(real(lam),  imag(eps))
show()
xlim(5,10)
plot(real(lam),  real(eps))
show()
