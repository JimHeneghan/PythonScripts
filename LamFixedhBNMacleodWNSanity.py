import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
#all units are in m
eps_inf = 4.87
s = 1.83
omega_nu = 137200
gamma = 3198.7#/(2*math.pi)#700.01702
d = 83e-9
imp0 = 376.730313
k0 = linspace (100000, 200000, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k0 - k0*k0)

n1 = np.sqrt(eps1)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0

# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*k0*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
# ylabel("T", fontsize = '30')   
ax.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
ax.legend(loc='upper right', fontsize='10')
# plt.tight_layout()
Lambda, R, T, RT = loadtxt("NewGamma7.5psRT5_22.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)
#R = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
plt.xlim(5,9)
plt.ylim(0,1)
print len(RT)
print min(RT[0:1000])
wn = (1/(Lambda*1e-6))/100
plt.plot(Lambda,R, label = r'$\rm R_{Lumerical:  Correct \ \gamma}$', color = "red", linewidth = 3)
plt.plot(Lambda,T, label = r'$\rm T_{Lumerical: Correct \ \gamma}$', color = "black", linewidth = 3)
plt.plot(Lambda,RT, label = r'$\rm R+T_{Lumerical: Correct \ \gamma}$', color = "limegreen", linewidth = 3)


plot(1/(k0*1e-6), Rm, label=r'$\rm R_{Macleod:  Correct \ \gamma}$', color='darkred')
plot(1/(k0*1e-6), Tm,  label=r'$\rm T_{Macleod: Correct \ \gamma}$', color='green')
plot(1/(k0*1e-6), Rm + Tm,  label=r'$\rm R+T_{Macleod: Correct \ \gamma}$', color='blue')

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center left', fontsize='30')
ax.axvline(x = 7.295, color = 'black', linewidth = 2)
# plt.savefig("hBNRealGamma250psMacConfirmLam.pdf")
# plt.savefig("hBNRealGamma250psMacConfirmLam.png")

plt.show()
