#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *
import harminv
import numpy.testing as nt

ff = open("HImodes.txt", 'w')
for z in range (1,68):
    E1 = "EmptyLat%d" %z
    E = E1 + '.txt'
    print E
    i = z
    E0R = loadtxt(E, usecols=(1,), skiprows= 1, unpack =True)
    E0I = loadtxt(E, usecols=(2,), skiprows= 1, unpack =True)
    E0  = np.zeros(len(E0R), dtype=np.complex64)
    for m in range (0, len(E0R)):
            E0[m] = E0R[m] + 1j*E0I[m]
    M = len(E0R)
    E0chop = E0[1500: M]
    E0_inv = harminv.Harminv(E0chop, fmin = 0, fmax = 0.015, nf = 300, dt = 1.66667e-11)

     for k in range (0, 1000):
     	 L[k] = (1/(2*math.pi))*((1/(2*math.pi*E0_inv.decay[0]))/((domega*k - E0_inv.freq[0]) + (1/(4*math.pi*E0_inv.decay[0]))**2))


    plot(L)
    show()
    for k in range (0, E0_inv.freq.size):
            thang1 = str(i)
            thang2 = str(E0_inv.freq[k])
            ff.write(thang1)
            ff.write("\t")
            ff.write(thang2)
            ff.write("\n")
ff.close()
