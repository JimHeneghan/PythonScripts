#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *
E1 = "Baby0" 
E = E1 + '.txt'
print E
R = np.zeros(10000, dtype=np.complex64)
T = np.zeros(10000, dtype=np.complex64)

E0R = loadtxt(E, usecols=(3,), skiprows= 1, unpack =True)

E0I = loadtxt(E, usecols=(4,), skiprows= 1, unpack =True)


for m in range (0, len(E0R)):
    R[m] = E0R[m] + 1j*E0I[m]

for i in range(0, 3000):
    I = R[:3000]
for i in range(3000, len(R)):
    I = append(I,[0])
Ref = R[3000:8000]
for i in range (0, 3000):
    Ref = insert(Ref, 1, [0])
for i in range (8000, len(R)):
    Ref = append(Ref, [0])
plot(R)
show()
plot(Ref)
show()

plot(I)
show()

    
dt = 0.125e-12

fs = 1/dt
f = fs*arange(0, len(E0R))/len(E0R)
R_fft = fft(Ref,len(E0R))
I_fft = fft(I,len(E0R))

plot(f, I_fft)
show()
plot(f, 20*np.log10((R_fft/I_fft)))

#xlim(0, 100e9)
show()




