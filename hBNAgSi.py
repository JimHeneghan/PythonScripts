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
# ax1.set_xlim(5,9)
plt.rc('axes', linewidth=2) 

q = 0
m = 0

mpl.rcParams['agg.path.chunksize'] = 10000
# all units are in rad/s

eps_inf = 4.87
s = 1.83
omega_nu = 2.58276e14
gamma = 6.02526055e12#/(2*math.pi)#700.01702
d_hBN = 80e-9
imp0 = 376.730313
w = linspace (180e12, 5e15,  1500000)
c0 = 3e8
eps_hBN = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w - w*w)

n_hBN = np.sqrt(eps_hBN)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k00 = 0

# unlabled equation on p 38 in Macleod after eqn 2.88 
delta_hBN = n_hBN*d_hBN*(w/c0)

################################################################
########################    Ag TM    ###########################

wp_Ag = 1.15136316e16
# w = linspace (100e12, 3600e12,  100)
eps_inf_Ag = 1.0
eps_s_Ag   = 343.744
tau_Ag     = 7.90279e-15
sigma_Ag   = 3.32069e6
eps0 = 8.85418782e-12

eps_Ag = eps_inf + (eps_s_Ag-eps_inf_Ag)/(1+1j*w*tau_Ag) + sigma_Ag/(1j*w*eps0)

n_Ag = np.sqrt(eps_Ag)

d_Ag = 50e-9
# imp0 = 376.730313
# k0 = k*1e12
# c0 = 3e8
# dx = 50e-9
# dy = 50e-9
nref = 1.0
n_Si = math.sqrt(11.7)
#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
delta_Ag = n_Ag*d_Ag*(w/c0)#*2*math.pi
################################################################

# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n_hBN)*imp0
eta2 = (n_Ag)*imp0
eta3 = (n_Si)*imp0

B11 = (cos(delta_hBN)*cos(delta_Ag)) - (eta2*sin(delta_hBN)*sin(delta_Ag)/eta1)
B12 = (1j*cos(delta_hBN)*sin(delta_Ag)/eta2) + (1j*sin(delta_hBN)*cos(delta_Ag)/eta1)

C11 = (1j*eta1*sin(delta_hBN)*cos(delta_Ag)) + (1j*eta2*cos(delta_hBN)*sin(delta_Ag))
C12 = -1*(eta1*sin(delta_hBN)*sin(delta_Ag)/eta2) + (cos(delta_hBN)*cos(delta_Ag))

B = B11 + B12*eta3
C = C11 + C12*eta3

Y = C/B

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

Tm = 4*eta0*real(eta3)/((eta0*B + C)*conj(eta0*B + C))


#plt.plot(k, R_exp, label = "grabbed data %d nm" %thick[z])
#plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$ ""\n $\chi^{2} = %g$ \n $\\theta = %g^{\circ}$ \n hBN is %d nm thick" %(gammaLow, chilow, inc*180/math.pi, thick[z]))
plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='red', linewidth = 3)
plt.plot((c0*2*math.pi/(w))*1e6, Tm, label=r'$\rm T_{TM}$', color='black', linewidth = 3)
plt.plot((c0*2*math.pi/(w))*1e6, 1-(Rm+Tm), label=r'$\rm A_{TM}$', color='limegreen', linewidth = 3)

# ax1.plot((1e6)/(k_data*100), R_exp, 'o',  markersize=3, markeredgecolor = "black", markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface

        

    #    legend(loc='lower center', fontsize='30')
ax1.set_ylabel("R", fontsize = '35')
ax1.set_xlabel(r'$\rm Wavelength \ (\mu m)$', fontsize = '35')
ax1.legend(loc='center right', fontsize='15')
ax1.axvline(x = 6.9, linestyle = "dashed", color = 'blue', linewidth = 2)
ax1.axvline(x = 7.5, linestyle = "dashed", color = 'red',  linewidth = 2)

plt.tight_layout()
plt.savefig("hBNAgSiR.png")
plt.savefig("hBNAgSiR.pdf")

################################################################

fig, ax1 = plt.subplots(1, figsize = (15,9))

plt.setp(ax1.spines.values(), linewidth=2)
ax1.tick_params(direction = 'in', width=2, labelsize=15)
ax1.tick_params(axis = 'x', direction = 'in', width=2, labelsize=15)
# ax1.set_xlim(5,9)
plt.rc('axes', linewidth=2) 
ax1.set_xlim(5,9)

plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='red', linewidth = 3)
plt.plot((c0*2*math.pi/(w))*1e6, Tm, label=r'$\rm T_{TM}$', color='black', linewidth = 3)
plt.plot((c0*2*math.pi/(w))*1e6, 1-(Rm+Tm), label=r'$\rm A_{TM}$', color='limegreen', linewidth = 3)

# ax1.plot((1e6)/(k_data*100), R_exp, 'o',  markersize=3, markeredgecolor = "black", markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface

        

    #    legend(loc='lower center', fontsize='30')
ax1.set_ylabel("R", fontsize = '35')
ax1.set_xlabel(r'$\rm Wavelength \ (\mu m)$', fontsize = '35')
ax1.legend(loc='center right', fontsize='15')
ax1.axvline(x = 6.9, linestyle = "dashed", color = 'blue', linewidth = 2)
ax1.axvline(x = 7.5, linestyle = "dashed", color = 'red',  linewidth = 2)

plt.tight_layout()
plt.savefig("hBNAgSiR5_9.png")
plt.savefig("hBNAgSiR5_9.pdf")
