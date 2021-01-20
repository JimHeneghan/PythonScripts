from numpy import *
from scipy import *
from pylab import *
import math

k = linspace(50000, 200000, 2000)
#Jiang et al values for Eps Parallel 

c = 3e8
eps_infzJ = 2.95
S_nuzJ = 0.61
omega_nuzJ = (1.41e14)/(2*math.pi*c)
gamma_nuzJ = (3.97e12)/(2*math.pi*c)

epskzJ = eps_infzJ + S_nuzJ*((omega_nuzJ**2)/(omega_nuzJ**2-1j*gamma_nuzJ*k - k**2))

## ZZ calculation
omega_TOz = 78000
omega_LOz = 83000
gamma_nuzZZ = 400
eps_infzZZ = 2.95

epskzZZ = eps_infzZZ*( 1 + (omega_LOz**2 - omega_TOz**2)/(omega_TOz**2-1j*gamma_nuzZZ*k - k**2))

plot(k/100, imag(epskzZZ),  label = r"$\epsilon_{\parallel}$ ZZ")
plot(k/100, imag(epskzJ),   label = r"$\epsilon_{\parallel} $ J")
ylabel(r"$Im(\epsilon)$", fontsize = '30')
xlabel(r"$cm^{-1}$", fontsize = '30')
legend(loc= 'upper left', fontsize='20')
show()
