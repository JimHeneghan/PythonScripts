#hi from the outside
import numpy as np
from scipy import *
from pylab import *
import math
from ctypes import *

lam = loadtxt("hBN_Eps_Im.txt", usecols=(0,),  unpack =True)
ImEps = loadtxt("hBN_Eps_Im.txt", usecols=(1,),  unpack =True)
ReEps = loadtxt("hBN_Eps_Real.txt", usecols=(1,),  unpack =True)

alpha_2a = np.zeros(len(lam), dtype=np.complex64)
#alpha_2b = np.zeros(len(lam), dtype=np.complex64)

eps0 = 8.85418782e-12
C1 = math.sqrt(2)*2*math.pi/(lam*1e-6)
Re = ReEps#/eps0
Im = ImEps#/eps0
alpha_2a =C1*np.sqrt((Re  + np.sqrt(Re*Re + Im*Im))) 


xlim(5,10)
ylabel("$m^-1$")
xlabel("$um$")
plot(lam, alpha_2a)

show()
