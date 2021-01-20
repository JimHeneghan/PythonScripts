import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *

R_exp, k = loadtxt("hBN60nmRef.txt", usecols=(0,1,),  unpack = True)

eps_inf = 4.87
s = 1.83
omega_nu = 1372
c = 3e10
d = 60e-7
imp0 = 376.730313
chi2 = 20
gamma = [7.017, 52.43]
for i in range (0, 2):
    eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma[i]*k - k*k)
    n1 = np.sqrt(eps1)
    delta1 = n1*d*k*2*math.pi
    
    eta0 = imp0
    eta1 = (n1)*imp0
    eta2 = imp0
    
    Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))
    R = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

    chi1 = sum((R_exp - R)**2)
    if (chi1 < chi2):
        gammaLow = gamma[i]
        lowi = i
        chilow = chi1
        Rlow = R
        #print "true"
        #print "for gamma = %g" %gamma[i]
        #print "chi is %g" %chi1
        #print i
    chi2 = chi1
    if (i == 0):
        RDamaged = R

print "gamma low is %g" %gammaLow
print "i is %g" %lowi
print  "Chi low is %g" %chilow

rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
ylabel("R", fontsize = '30')
xlabel(r'$\lambda (\mu m)$', fontsize = '30')
plt.plot(k, RDamaged, label = "Damaged")
plt.plot(k, R, label = "Clean")
plt.plot(k, R_exp, label = "grabbed data 60 nm")
#plt.plot(k, Rlow, label = r"fitted gamma is %g" %gammaLow)
legend(loc='upper right', fontsize='10')
plt.tight_layout()
plt.show()
