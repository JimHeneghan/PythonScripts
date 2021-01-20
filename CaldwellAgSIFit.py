import numpy as np
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy import *
from scipy.optimize import curve_fit
from pylab import *

thick  = [60, 105, 210, 490, 1020, 1990, 6400]
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
gam    = [40.047, 27.2056, 9.622, 17.4781, 8.7441, 9.0982, 1.3777]

fig, axs = plt.subplots(2,2, figsize = (15,9))


    # ax.set_xlim(5,9)
# ax.set_xlim(5,9)

z = 0
q = 0
m = 0
for ax in axs.flat:  
    plt.setp(ax.spines.values(), linewidth=2)
    ax.tick_params(direction = 'in', width=2, labelsize=15)
    ax.tick_params(axis = 'x', direction = 'in', width=2, labelsize=15)

    plt.rc('axes', linewidth=2) 
    # z = 3-q
    print(z) 

    F1 = "hBN%dnmRef.txt" %thick[z]
    k_data, R_exp = loadtxt(F1, usecols=(0,1,),  unpack = True)
    c0 = 3e8

    eps_inf = 4.87
    s = 1.83
    omega_nu = 2.58276e14
    gamma = (100*c0*gam[z])/2*math.pi#/()#700.01702
    d_hBN = thick[z]*1e-9

    wp_Ag = 1.15136316e16
    eps_inf_Ag = 1.0
    eps_s_Ag   = 343.744
    tau_Ag     = 7.90279e-15
    sigma_Ag   = 3.32069e6
    eps0 = 8.85418782e-12
    d_Ag = 50e-9

    imp0 = 376.730313
    w = linspace (180e12, 3e15,  1500000)


    #using the equations in chapter 2.2.r of Macleod
    #assuming the impedance of free space cancels out
    #assuming the incident media is vacuum with k00 = 0

    # unlabled equation on p 38 in Macleod after eqn 2.88 

    # chi2 = 20
    # inc = math.pi*0.0/180.0
    
    # # nb = np.zeros(len(k_data), dtype=np.complex64)
    #gamma = linspace (1, 300, 100001)

    eps_Ag = eps_inf + (eps_s_Ag-eps_inf_Ag)/(1+1j*w*tau_Ag) + sigma_Ag/(1j*w*eps0)
    n_Ag = np.sqrt(eps_Ag)

    eps_hBN = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w - w*w)
    
    n_Si = math.sqrt(11.7)
    n_hBN = np.sqrt(eps_hBN)
    
    delta_hBN = n_hBN*d_hBN*(w/c0)
    delta_Ag = n_Ag*d_Ag*(w/c0)#*2*math.pi

    eta0 = imp0
    eta1 = (n_hBN)*imp0
    eta2 = imp0#(n_Ag)*imp0
    eta3 = imp0#(n_Si)*imp0
    print(delta_Ag)

    B11 = (cos(delta_hBN)*cos(delta_Ag)) - (eta2*sin(delta_hBN)*sin(delta_Ag)/eta1)
    B12 = (1j*cos(delta_hBN)*sin(delta_Ag)/eta2) + (1j*sin(delta_hBN)*cos(delta_Ag)/eta1)

    C11 = (1j*eta1*sin(delta_hBN)*cos(delta_Ag)) + (1j*eta2*cos(delta_hBN)*sin(delta_Ag))
    C12 = -1*(eta1*sin(delta_hBN)*sin(delta_Ag)/eta2) + (cos(delta_hBN)*cos(delta_Ag))

    B = B11 + B12*eta3
    C = C11 + C12*eta3

    Y = C/B

    Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

    Tm = 4*eta0*real(eta3)/((eta0*B + C)*conj(eta0*B + C))


    ax.plot((c0*2*math.pi/(w))*1e6, 1-(Rm+Tm), label=r'$\rm A_{TM \ d = %.2f}$' %thick[z], color=shades[z], linewidth = 3)
    # plt.plot((c0*2*math.pi/(w))*1e6, Tm, label=r'$\rm T_{TM}$', color='black', linewidth = 3)
    # plt.plot((c0*2*math.pi/(w))*1e6, 1-(Rm+Tm), label=r'$\rm A_{TM}$', color='limegreen', linewidth = 3)

        # return abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y))*J)

    ax.set_ylabel("A", fontsize = '35')
    ax.set_xlabel(r'$\rm Wavelength \ (\mu m)$', fontsize = '35')
    ax.legend(loc='center right', fontsize='15')

    # ax.set_xlim(7.25,7.5)
    # ax.set_ylim(0,0.3)


    ax.axvline(x = 6.9, linestyle = "dashed", color = 'blue', linewidth = 2)
    ax.axvline(x = 7.5, linestyle = "dashed", color = 'red',  linewidth = 2)
    z = z + 1

plt.tight_layout()
plt.savefig("ACaldwelAgSiVac.png")# %thick[z])
plt.savefig("ACaldwelAgSiVac.pdf")# %thick[z])




################################################################

# fig, ax = plt.subplots(1, figsize = (15,9))

# plt.setp(ax.spines.values(), linewidth=2)
# ax.tick_params(direction = 'in', width=2, labelsize=15)
# ax.tick_params(axis = 'x', direction = 'in', width=2, labelsize=15)
# # ax.set_xlim(5,9)
# plt.rc('axes', linewidth=2) 
# ax.set_xlim(5,9)

# # plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='red', linewidth = 3)
# # plt.plot((c0*2*math.pi/(w))*1e6, Tm, label=r'$\rm T_{TM}$', color='black', linewidth = 3)
# # plt.plot((c0*2*math.pi/(w))*1e6, 1-(Rm+Tm), label=r'$\rm A_{TM}$', color='limegreen', linewidth = 3)

# # ax.plot((1e6)/(k_data*100), R_exp, 'o',  markersize=3, markeredgecolor = "black", markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface


#     #    legend(loc='lower center', fontsize='30')
# ax.set_ylabel("R", fontsize = '35')
# ax.set_xlabel(r'$\rm Wavelength \ (\mu m)$', fontsize = '35')
# ax.legend(loc='center right', fontsize='15')
# # ax.axvline(x = 6.9, linestyle = "dashed", color = 'blue', linewidth = 2)
# # ax.axvline(x = 7.5, linestyle = "dashed", color = 'red',  linewidth = 2)

# plt.tight_layout()
# plt.savefig("CaldwelonAgSi.png")
# plt.savefig("CaldwelonAgSi.pdf")

################################################################################
################################################################################
################################################################################

