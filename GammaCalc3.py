import numpy as np
from scipy import *
from pylab import *
import math
#all units are in cm
eps_inf = 4.87
s = 1.83
omega_nu = 1372
gamma =4.01702
c = 3e10
d = 83e-7
k = linspace (1000, 2000, 20000)

eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) - 1j*gamma*k - k*k)

n1 = np.sqrt(eps1)

# defining the constants used in eqn(5.4.2) of Orfanidis assuming a layer of
# 83 nm thick hBN in a vacuum on either side 
eta0 = 1
eta1 = 1/n1
eta2 = 1
k1 = k*n1

ro1 = (eta1 - eta0)/(eta1 + eta0)

ro2 = (eta2 - eta1)/(eta2 + eta1)

R = abs((ro1 + ro2*np.exp(2*1j*k1*d))/(1 + ro1*ro2*np.exp(2*1j*k1*d)))

# equation from problem 5.11 in Orfanidis
T=abs(1.0/(cos(k1*d)-1j*(n1+1.0/n1)*sin(k1*d)/2.0))

#plot(k, abs(R))
ylim(0, 1)
xlim(5, 10)
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
plot(1e4/k,R, label = 'R', color = 'red')
plot(1e4/k,T, label = 'T', Color = 'blue')
legend(loc='upper left', fontsize='20')
#plot(k, real(eps1))
show()

