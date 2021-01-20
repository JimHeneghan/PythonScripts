import numpy as np
from scipy import *
from pylab import *
import math
#all units are in cm
eps_inf = 4.87
s = 1.83
omega_nu = 1372
c = 3e10
d = 83e-7
imp0 = 376.730313

def model
#k = linspace (1000, 2000, 200000)
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k0 - k0*k0)
n1 = np.sqrt(eps1)

delta1 = n1*d*k0*2*math.pi

eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

R = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

