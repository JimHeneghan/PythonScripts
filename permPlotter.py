#hi from the outside
from numpy import *
from scipy import *
from pylab import *
import math

lam = loadtxt("hBNgetfdtdindexMO5anal3.txt", usecols=(0,),  unpack =True)
n_r = loadtxt("hBNgetfdtdindexMO5anal3.txt", usecols=(1,),  unpack =True)
n_i = loadtxt("hBNgetfdtdindexMO5anal3.txt", usecols=(2,),  unpack =True)

nc = n_r + 1j*n_i
eps = nc**2

xlim(5,10)
plot(real(lam), real(eps))
show()
