#hi from the outside
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *

#lam = np.zeros(2000, dtype=np.complex64)
n =  np.zeros(2000, dtype=np.complex64)
LossTan = np.zeros(2000, dtype=np.complex64)
lam = loadtxt("hBNgetmaterial.txt", usecols=(0,),  unpack =True)
n_r = loadtxt("hBNgetmaterial.txt", usecols=(1,),  unpack =True)
n_i = loadtxt("hBNgetmaterial.txt", usecols=(2,),  unpack =True)

for m in range (0, len(n_r)):
    n[m] = n_r[m] + 1j*n_i[m]
c = 3e8
LossTan = n_i/n_r
alpha = math.pi*LossTan*n/(lam/c)


#alpha_2b = C1*np.sqrt((Re - np.sqrt(Re*Re - Im)))
#xlim(5,10)
plot(real(lam/c), alpha)
#show()
#plot(lam, alpha_2b)
show()
