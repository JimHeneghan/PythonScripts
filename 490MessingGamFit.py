import numpy as np
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy import *
from scipy.optimize import curve_fit
from pylab import *

thick = [60, 105, 210, 490, 1020, 1990, 6400]

fig, axs = plt.subplots(figsize = (19, 9),constrained_layout=True)
rc('axes', linewidth=2)

for z in range (3,4):   
    F1 = "hBN%dnmRef.txt" %thick[z]
    k_data, R_exp = loadtxt(F1, usecols=(0,1,),  unpack = True)
    eps_inf = 4.87
    s = 1.83
    omega_nu = 1372
    d = thick[z]*1e-7
    imp0 = 376.730313
    chi2 = 20
    inc = math.pi*25.0/180.0
    
    nb = np.zeros(len(k_data), dtype=np.complex64)
    def func(k, gamma, J):
        lam = (1/(k*100))*1e6
        nb = np.sqrt(1+ 0.33973 + ((0.81070 * (lam**2))/(lam**2 - 0.10065**2))
                         + ((0.19652*(lam**2))/(lam**2 - 29.87**2)) + ((4.52469*(lam**2))/(lam**2 - 53.82**2)))
        eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k - k*k)
        n1 = np.sqrt(eps1)
        delta1 = n1*d*k*2*math.pi*cos(inc)
        
        eta0 = imp0
        eta1 = (n1)*imp0
        eta2 = imp0*nb
        
        Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))
        
        return abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y))*J)

    popt, pcov = curve_fit(func, k_data, R_exp, bounds = (0, [300.0, 1.0]))
    print popt
    axs.plot(k_data, func(k_data, *popt), 'g--', linewidth = 3, label = r"$\rm \gamma_{fit} = %.2f$ " %popt[0])# "\n" r"J = %f" %tuple(popt))
    print"thickness = %d \t gamma = %0.2f \t J = %0.2f " %(thick[z], popt[0], popt[1])
    axs.plot(k_data, R_exp, linewidth = 3)#, label = "Digitized Data")
    plt.setp(axs.spines.values(), linewidth=2)
    axs.set_ylabel("R", fontsize = '30')
    axs.set_ylim(0,1)
    axs.set_title("d= %d nm" %thick[z], fontsize = '25')
    axs.tick_params(width=2, labelsize=20)

    axs.legend(fancybox = True, loc='upper left', fontsize='17', framealpha=1)

plt.savefig("490CurveFit.png")

plt.show()
