#hi from the outside
import numpy as np
from scipy import *
from pylab import *
import math

lam2 = loadtxt("hBNgetindexConstZ.txt", usecols=(0,),  unpack =True)
n_i = loadtxt("hBNgetindexConstZ.txt", usecols=(2,),  unpack =True)
n_r = loadtxt("hBNgetindexConstZ.txt", usecols=(1,),  unpack =True)

C1 = math.sqrt(2)*2*math.pi/(lam2*1e-6)
nc = n_r + 1j*n_i
eps = nc**2
ReEps = real(eps)
ImEps = imag(eps)
alpha_2a =C1*np.sqrt((np.sqrt(ReEps*ReEps + ImEps*ImEps) - ReEps))

lam1 = loadtxt("hBN_RTConstZ32.txt", usecols=(0,),  unpack =True)
T = loadtxt("hBN_RTConstZ32.txt", usecols=(2,),  unpack =True)
R = loadtxt("hBN_RTConstZ32.txt", usecols=(1,),  unpack =True)

d = 80e-9
alpha_1 = (-log(T/(1-R)))/d

xlim(5,10)
ylabel(r"$m^{-1}$", fontsize = '30')
xlabel("$um$", fontsize = '30')

plot(lam1, alpha_1,  label=r'$\alpha_1$')
plot(lam2, alpha_2a, label=r'$\alpha_2$')
legend(loc='upper left')

show()
