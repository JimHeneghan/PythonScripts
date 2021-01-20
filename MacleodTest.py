import numpy as np
from scipy import *
from pylab import *
import math
#all units are in cm
eps_inf = 4.87
s = 1.83
omega_nu = 1372
gamma =7.01702
c = 3e10
d = 83e-7
imp0 = 376.730313
#k = linspace (1000, 2000, 200000)
lambda2, RFDTD= loadtxt("hBN_RTConstZ32.txt", usecols=(0,1,),  unpack = True)
#################################################################################

#################################################################################
k0 = 1/(lambda2*1.0e-4)
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

R = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))
#################################################################################

#################################################################################
#from the pulled data
lambda1, n_r, n_i = loadtxt("hBNgetindexConstZ.txt", usecols=(0,1,2),  unpack = True)

# Calculate Complex Refractive Index
nc = n_r - 1j*n_i
eps = nc**2

k02 = 1/(lambda1*1.0e-4)

delta2 = nc*d*k02*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta12 = (nc)*imp0
eta2 = imp0
Y2 =  (eta2*cos(delta2) + 1j*eta12*sin(delta2))/(cos(delta2) + 1j*(eta2/eta12)*sin(delta2))

R2 = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#plot(lambda2, real(eps1)) #sanity test function to test the calculated permittivity 
#show()
# Formatting and Plotting
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)

#xlim(400,2000)
ylabel("R", fontsize = '30')
xlabel(r'$\lambda (\mu m)$', fontsize = '30')

plot(lambda2, R, label=r'$R_{Macleod, Jiang \ \epsilon}$', color='red', linewidth = 6)
plot(lambda2, R2,  label=r'$R_{Macleod: getindex}$', color='limegreen', linewidth = 4)
plot(lambda2, RFDTD,  label=r'$R_{Lumerical: R }$', color='blue')
plt.tight_layout()
legend(loc='upper left', fontsize='15')

plt.show()
