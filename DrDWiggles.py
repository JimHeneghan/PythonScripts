#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt

dTime = (linspace(1, 100000))
dx = (1e-6)/25
je = 100.0
dt = dx/(3e8*2.0)
omega  = 2*2.0*3e8*math.pi/(je*dx)
arg = (dTime*dt - dt*10000.0)/50.0/dt
arg2 = arg*arg
y = -np.exp(-0.5*arg2)*np.cos(omega*(dTime*dt - 10000*dt))

plot(dTime*dt, y)
show()
#plot(fft(dTime*dt), abs(fft(y))**2)
#show()


