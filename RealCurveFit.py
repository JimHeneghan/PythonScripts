import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.optimize import curve_fit
from pylab import *

#thick = [60, 105, 210, 490, 1020, 1990, 6400]
thick = [6400]

for z in range (0, len(thick)):   
    F1 = "hBN%dnmRef.txt" %thick[z]
    k_data, R_exp = loadtxt(F1, usecols=(0,1,),  unpack = True)
    k_data = k_data[120:len(k_data)]
    R_exp = R_exp[120:len(R_exp)]
    eps_inf = 4.87
    s = 1.83
    omega_nu = 1372
    d = thick[z]*1e-7
    imp0 = 376.730313
    chi2 = 20
    inc = math.pi*25.0/180.0
    
    nb = np.zeros(len(k_data), dtype=np.complex64)
    #gamma = linspace (1, 300, 100001)
    def func(k, gamma):
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

        
        return abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

    popt, pcov = curve_fit(func, k_data, R_exp)
    print popt
    fig, ax = plt.subplots(figsize=(19,9))
    ax = plt.subplot(111)
    rc('axes', linewidth=2)
    tick_params(width=2, labelsize=20)
    ylabel("R", fontsize = '30')
    xlabel(r'$\rm Frequency \ (cm^{-1})$', fontsize = '30')

    #plt.plot(k, R_exp, label = "grabbed data %d nm" %thick[z])
    #plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$ ""\n $\chi^{2} = %g$ \n $\\theta = %g^{\circ}$ \n hBN is %d nm thick" %(gammaLow, chilow, inc*180/math.pi, thick[z]))
    plt.plot(k_data, func(k_data, *popt), 'g--', linewidth = 3, label = r"$\rm \gamma_{fit} = %g$ "" \n hBN Thickness = %g nm" %(popt, thick[z]))
    plt.plot(k_data, R_exp, linewidth = 3, label = "Digitized Data")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # if(z < 3):
    #     ax.legend(fancybox = True, loc='upper right', fontsize='25', framealpha=1)
    # else:
    ax.legend(fancybox = True, loc='lower center', fontsize='25', framealpha=1)
    #    legend(loc='lower center', fontsize='30')
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=2)
    #plt.savefig("CurveFit/hBNCurveFit%gnm.png" %thick[z])
    #plt.savefig("CurveFit/hBNCurveFit%gnm.pdf" %thick[z])
    #plt.cla()
    plt.show()
