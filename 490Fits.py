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

fig, axs = plt.subplots(3, 2, figsize = (16, 12),constrained_layout=True)
#axs = plt.subplot(111)
rc('axes', linewidth=2)



#plt.tight_layout()
#
q = 0
m = 0
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
    J = 1.0
    
    nb = np.zeros(len(k_data), dtype=np.complex64)
    #gamma = linspace (1, 300, 100001)
    def func(k, gamma):
        gamma = 9
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

        
        return J*abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

    popt, pcov = curve_fit(func, k_data, R_exp)#, bounds = (0, 300))
    print popt
    

    #plt.plot(k, R_exp, label = "grabbed data %d nm" %thick[z])
    #plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$ ""\n $\chi^{2} = %g$ \n $\\theta = %g^{\circ}$ \n hBN is %d nm thick" %(gammaLow, chilow, inc*180/math.pi, thick[z]))
    axs[q, m].plot(k_data, func(k_data, *popt), 'g--', linewidth = 3, label = r"$\rm \gamma_{fit} = %.2f$ " %popt[0])# "\n" r"J = %f" %tuple(popt))
    print"thickness = %d \t gamma = %0.2f \t J = %0.2f " %(thick[z], popt[0], J)
    axs[q, m].plot(k_data, R_exp, linewidth = 3)#, label = "Digitized Data")
    plt.setp(axs[q, m].spines.values(), linewidth=2)
    axs[q, m].set_ylabel("R", fontsize = '30')
    axs[q, m].set_ylim(0,1)
    axs[q, m].set_title("d= %d nm" %thick[z], fontsize = '25')
    axs[q, m].tick_params(width=2, labelsize=20)
    #box = ax.get_position()
    #axs[z].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #if(z < 3):
    #axs[q, m].legend(fancybox = True, loc='upper right', fontsize='20', framealpha=1)
    #else:
    axs[q, m].legend(fancybox = True, loc='upper left', fontsize='17', framealpha=1)
    if(z == 2 or z == 3 ):
        axs[q, m].legend(fancybox = True, loc='upper right', fontsize='17', framealpha=1)
    #if(m == 1):
        
    if (q == 2):
        axs[q, m].set_xlabel(r'$\rm Frequency \ (cm^{-1})$', fontsize = '30')
        m = 1
        q = -1
    q  = q + 1
    # if (m == 1):    
    #     q = 1
    #     m = 0

    # else:
    #     m = m + 1
        

    #    legend(loc='lower center', fontsize='30')

#plt.savefig("/halhome/jimheneghan/UsefulImages/hBNThicknessGammaFitFeb3.png")
#plt.savefig("/halhome/jimheneghan/UsefulImages/hBNThicknessGammaFitFeb3.pdf")
    #plt.cla()
plt.show()
