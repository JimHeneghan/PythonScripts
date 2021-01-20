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
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
gamma = [40.047, 27.2056, 9.622, 17.4781, 8.7441, 9.0982, 1.3777]
fig, ax1 = plt.subplots(1, figsize = (15,9))

plt.setp(ax1.spines.values(), linewidth=2)
ax1.tick_params(direction = 'in', width=2, labelsize=15)
ax1.tick_params(axis = 'x', direction = 'in', width=2, labelsize=15)
ax1.set_xlim(5,9)
plt.rc('axes', linewidth=2) 

q = 0
m = 0
for z in range (0, 6):   
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
    #gamma = linspace (1, 300, 100001)
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
    

    #plt.plot(k, R_exp, label = "grabbed data %d nm" %thick[z])
    #plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$ ""\n $\chi^{2} = %g$ \n $\\theta = %g^{\circ}$ \n hBN is %d nm thick" %(gammaLow, chilow, inc*180/math.pi, thick[z]))
    ax1.plot((1e6)/(k_data*100), func(k_data, *popt), linewidth = 3, color = shades[z], label = r"$d\rm_{hBN}  \ = \ %d \ nm$" %thick[z])# label = r"$\rm \gamma_{fit} = %.2f \ J = %.2f $ " %(popt[0], popt[1]))# "\n" r"J = %f" %tuple(popt))
    print"thickness = %d \t gamma = %0.2f \t J = %0.2f " %(thick[z], popt[0], popt[1])
    # ax1.plot((1e6)/(k_data*100), R_exp, 'o',  markersize=3, markeredgecolor = "black", markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface

        

    #    legend(loc='lower center', fontsize='30')
ax1.set_ylabel("R", fontsize = '35')
ax1.set_xlabel(r'$\rm Wavelength \ (\mu m)$', fontsize = '35')
ax1.legend(loc='center right', fontsize='15')
ax1.axvline(x = 6.9, linestyle = "dashed", color = 'blue', linewidth = 2)
ax1.axvline(x = 7.5, linestyle = "dashed", color = 'red',  linewidth = 2)

plt.tight_layout()
plt.savefig("CaldwelOnlyLines.png")
plt.savefig("CaldwelOnlyLines.pdf")

################################################################################
################################################################################
################################################################################

fig, ax1 = plt.subplots(1, figsize = (15,9))
plt.setp(ax1.spines.values(), linewidth=2)
ax1.tick_params(direction = 'in', width=2, labelsize=10)
ax1.tick_params(axis = 'x', direction = 'in', width=2, labelsize=0)
ax1.set_xlim(1200,1700)
plt.rc('axes', linewidth=2) 
#plt.tight_layout()
#
q = 0
m = 0
for z in range (0, 7):   
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
    #gamma = linspace (1, 300, 100001)
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
    

    #plt.plot(k, R_exp, label = "grabbed data %d nm" %thick[z])
    #plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$ ""\n $\chi^{2} = %g$ \n $\\theta = %g^{\circ}$ \n hBN is %d nm thick" %(gammaLow, chilow, inc*180/math.pi, thick[z]))
    ax1.plot(k_data, func(k_data, *popt), linewidth = 2, color = shades[z], label = r"$\rm d_{hBN}  \ = \ %d \ nm$" %thick[z])# label = r"$\rm \gamma_{fit} = %.2f \ J = %.2f $ " %(popt[0], popt[1]))# "\n" r"J = %f" %tuple(popt))
    print"thickness = %d \t gamma = %0.2f \t J = %0.2f " %(thick[z], popt[0], popt[1])
    ax1.plot(k_data, R_exp, 'o',  markersize=3, markeredgecolor = "black", markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface

        

    #    legend(loc='lower center', fontsize='30')
ax1.set_ylabel("R", fontsize = '15')
ax1.set_xlabel(r'$\rm Wavenumber \ (cm^{-1})$', fontsize = '15')
ax1.legend(loc='upper right', fontsize='10')
ax1.axvline(x = 1370, linestyle = "dashed", color = 'black', linewidth = 2)
ax1.axvline(x = 1610, linestyle = "dashed", color = 'black', linewidth = 2)

#