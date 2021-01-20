#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *

omega_p = 314e9
nu_c = 2.0e10
omega_b = 3e11

omega = np.linspace(1e10, 1e15, 1000000)

fig, ax = plt.subplots(figsize=(12,9))
#plt.tight_layout(pad=5., w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
ylabel(r"$\rm Re(\epsilon_{zz})$", fontsize = '30', labelpad = .05)   
xlabel(r"$\rm \omega \ (rad \ s^{-1})$", fontsize = '30')

# Epsxx = 1 - ((omega_p**2)*(omega - 1j*nu_c))/(omega*(((omega - 1j*nu_c)**2) - omega_b**2))

# plot(omega, real(Epsxx))
# show()

# xlim(0, 1e8)
# plot(omega, imag(Epsxx))
# show()

# Epsxy = ((omega_p**2)*(omega_b ))/(omega*((omega - 1j*nu_c)**2 - omega_b**2))

# plot(omega, real(Epsxy))
# show()

# #xlim(0, 1e8)
# plot(omega, imag(Epsxy))
#show()

Epszz = 1 - (omega_p**2)/omega*(omega - 1j*nu_c)
print min(real(Epszz))
print max(real(Epszz))

ylim(min(real(Epszz)), max(real(Epszz)))
plot(omega, real(Epszz))
show()

plt.savefig("ReEpsZZ.pdf")
plt.savefig("ReEpsZZ.png")

fig, ax = plt.subplots(figsize=(12,9))
#plt.tight_layout(pad=5., w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
ylabel(r"$\rm Im(\epsilon_{zz})$", fontsize = '30', labelpad = .05)   
xlabel(r"$\rm \omega \ (rad \ s^{-1})$", fontsize = '30')

ylim(0, max(imag(Epszz)))
plot(omega, imag(Epszz))
#show()

#plt.savefig("ImEpsZZ.pdf")
#plt.savefig("ImEpsZZ.png")
