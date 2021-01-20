import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *

thick = [60, 105, 210, 490, 1020, 1990, 6400]
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
rc('axes', linewidth=2) 
for z in range (0, len(thick)):   
    F1 = "hBN%dnmRef.txt" %thick[z]
    k, R_exp = loadtxt(F1, usecols=(0,1,),  unpack = True)
    eps_inf = 4.87
    s = 1.83
    omega_nu = 1372
    d = thick[z]*1e-7
    imp0 = 376.730313
    chi2 = 20
    inc = 0#math.pi*25.0/180.0
    lam = (1/(k*100))*1e6
    nb = np.zeros(len(k), dtype=np.complex64)
    nb = np.sqrt(1+ 0.33973 + ((0.81070 * (lam**2))/(lam**2 - 0.10065**2))
                     + ((0.19652*(lam**2))/(lam**2 - 29.87**2)) + ((4.52469*(lam**2))/(lam**2 - 53.82**2)))

    gamma = linspace (1, 300, 100001)
    for i in range (0, len(gamma)):
        eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma[i]*k - k*k)
        n1 = np.sqrt(eps1)
        delta1 = n1*d*k*2*math.pi*cos(inc)
        
        eta0 = imp0
        eta1 = (n1)*imp0
        eta2 = imp0#*nb
        
        Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))
        R = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

        chi1 = sum((R_exp - R)**2)

        if (chi1 < chi2):
            gammaLow = gamma[i]
            lowi = i
            chilow = chi1
            Rlow = R
    #        print "true"
    #        print "for gamma = %g" %gamma[i]
    #        print "chi is %g" %chi1
    #        print i
        chi2 = chi1


    print "gamma low is %g" %gammaLow
    print "i is %g" %lowi
    print  "Chi low is %g" %chilow

    # rc('axes', linewidth=2)
    # tick_params(width=2, labelsize=20)
    ylabel("R", fontsize = '30')
    xlabel(r'$\rm Wavenumber \ (cm^{-1})$', fontsize = '30')

    #plt.plot(k, R_exp, label = "grabbed data %d nm" %thick[z])
    plt.plot(k, Rlow, linewidth = 2, color = shades[z], label = r"$\rm \ hBN \ thicknes \ is \ %d \ nm \ thick$" %thick[z])
    # legend(loc='upper right', fontsize='10')
    
ax.legend(loc='upper right', fontsize='20')
axvline(x =1/(7.29*1e-4), color = 'black')
plt.tight_layout()

plt.savefig("hBNAllThickBaF2Normal_WN.png")
plt.savefig("hBNAllThickBaF2Normal_WN.pdf")
plt.show()
