from numpy import *
from scipy import *
from pylab import *
import math

#in k
eps_inf = 4.87
S_nu = 1.83
c = 3e8
omega_nu = (2.58e14)/(2*math.pi*c)
gamma_nu = (1.32e12)/(2*math.pi*c)
gamma_nuZZ = 500

print omega_nu/100
print gamma_nu/100


k = linspace(50000, 200000, 2000)
##perpindicular calculation 

epsk = eps_inf + S_nu*((omega_nu**2)/(omega_nu**2-1j*gamma_nu*k - k**2))

epskZZ = eps_inf + S_nu*((omega_nu**2)/(omega_nu**2-1j*gamma_nuZZ*k - k**2))


##parallel calculation
eps_infz = 2.95
S_nuz = 0.61
omega_nuz = (1.41e14)/(2*math.pi*c)
gamma_nuz = (3.97e11)/(2*math.pi*c)

epskz = eps_infz + S_nuz*((omega_nuz**2)/(omega_nuz**2-1j*gamma_nuz*k - k**2))

gamma_nuz2 = 400
epskz2 = eps_infz + S_nuz*((omega_nuz**2)/(omega_nuz**2-1j*gamma_nuz2*k - k**2))

plot(1e6/k, real(epsk), color = "black", label = r"$\epsilon_{\perp}$ Jiang")
plot(1e6/k, real(epskZZ),  label = r"$\epsilon_{\perp}$ Zhao")
plot(1e6/k, real(epskz), color = "black",label = r"$\epsilon_{\parallel}$ Jiang")
plot(1e6/k, real(epskz2), label = r"$\epsilon_{\parallel}$ Zhao")
ylabel(r"Re$(\epsilon)$", fontsize = '30')
xlabel(r"Wavelength $\mu m$", fontsize = '30')
legend(loc= 'upper left', fontsize='20')
show()

#plot(k/100, imag(epsk),  label = r"$\epsilon_{\perp}$")
#plot(k/100, imag(epskZZ),  label = r"$\epsilon_{\perp}$ ZZ")
plot(k/100, imag(epskz), label = r"$\epsilon_{\parallel}$")
plot(k/100, imag(epskz2), label = r"$\epsilon_{\parallel}$ ZZ")
ylabel(r"Im$(\epsilon)$", fontsize = '30')
xlabel(r"$cm^{-1}$", fontsize = '30')
legend(loc= 'upper left', fontsize='20')
show()

plot(k/100, real(epskz), label = r"$\epsilon_{\parallel}$")
plot(k/100, real(epskz2), label = r"$\epsilon_{\parallel}$ ZZ")
ylabel(r"Im$(\epsilon)$", fontsize = '30')
xlabel(r"$cm^{-1}$", fontsize = '30')
legend(loc= 'upper left', fontsize='20')
show()
