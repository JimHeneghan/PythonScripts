import numpy as np
from scipy import *
from pylab import *
import math
#all units are in m
eps_inf = 4.87
s = 1.83
omega_nu = 137200
gamma = 3198.5/6.28#700.01702
d = 83e-9
imp0 = 376.730313
k0 = linspace (100000, 200000, 20000)

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

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
T = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))


rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
xlim(1200,1700)
ylim(0,1)
ylabel("R", fontsize = '30')
xlabel(r'$\lambda (\mu m)$', fontsize = '30')

plot(k0/100, R, label=r'$R_{Macleod, Jiang_{\epsilon}}$', color='red')
plot(k0/100, T,  label=r'$T_{Macleod: Jiang_{\epsilon}}$', color='limegreen')
plot(k0/100, R + T,  label=r'$R + T}}$', color='blue')

plt.tight_layout()
legend(loc='center right', fontsize='15')

plt.show()
