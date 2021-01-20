import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *

#k, R_exp = loadtxt("hBN6400nmRef.txt", usecols=(0,1,),  unpack = True)
k = linspace(1200, 1700, 1000)
eps_inf = 4.87
s = 1.83
omega_nu = 1372
c = 3e10
d =[60e-7, 105e-7, 210e-7, 490e-7, 1990e-7, 6400e-7]
imp0 = 376.730313
gamma = [84.045, 32.018, 19.6306, 22.6297, 17.4973, 1.00989]

fig, ax = plt.subplots(figsize=(12,9))

plt.tight_layout()

for i in range (0, len(d)):
    eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma[i]*k - k*k)
    n1 = np.sqrt(eps1)
    delta1 = n1*d[i]*k*2*math.pi
    
    eta0 = imp0
    eta1 = (n1)*imp0
    eta2 = imp0
    
    Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))
    R = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

    #ax.plot(k, R, label = "hBN thickness %g nm" %(d[i]*1e7), zorder=(len(d) -i))
    


rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
xlabel("hBN Layer Thickness (nm)", fontsize = '30')
ylabel(r'$\rm \gamma (cm^{-1})$', fontsize = '30')


#plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$" %gammaLow)
ax.legend(loc='lower left', fontsize='10')
#ax.legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True, borderaxespad=7)
plt.tight_layout()
#fig.savefig('samplefigure', bbox_inches='tight')

#plt.show()
d1 = [60, 105, 210, 490, 1990, 6400]

base =  linspace(1, 1000000, 10000)
Amp = linspace(1, 100000, 10000)
y = np.zeros(6)
chi2 = 300000
for j in range (1, len(base)):
    for k in range (0, len(Amp)):
        for i in range (0, len(d)):
            y[i] = Amp[k]*exp(-base[j]*d[i])
            #ax.plot(d, y, label = "base is %g" %base[j]) 
        chi1 = sum((gamma - y)**2)
        if (chi1 < chi2):
            ylow = y
            baselow = j
            Amplow = k
print chi1
print base[baselow]
print Amp[Amplow]

ax.plot(d1, y, label = "base is %g" %base[baselow])      
ax.plot(d1, gamma, 'o')
ax.legend(loc='lower left', fontsize='10')
plt.show()
