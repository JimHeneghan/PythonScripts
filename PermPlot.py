import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams["font.family"] = "Times New Roman"

mpl.rcParams['agg.path.chunksize'] = 10000
# all units are in THz

eps_inf = 4.87
s = 1.83
omega_nu = 41.14e12
gamma = 0.95895e12
w0 = linspace (15e12, 300e12, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w0 - w0*w0)

Re_eps = np.real(eps1)
Im_eps = abs(np.imag(eps1))
lam    = (c0/w0)*1e6

lam_res = 7.29e-6
w_res   = c0/lam_res
eps_res = np.real(eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w_res - w_res*w_res))
#################################################################

eps_inf = 2.95
s = 0.61
omega_nu = 22.36640e12
gamma = 0.06045e12

eps1z = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w0 - w0*w0)

Re_epsz = np.real(eps1z)
Im_epsz = abs(np.imag(eps1z))
###############################################################################
############################## Peak Finding ###################################

# for i in range(0,len(Re_eps)-1):
#     if ((Re_eps[i] > Re_eps[i-1]) & (Re_eps[i] > Re_eps[i+1]) & (lam[i]<10)):
#     	peak  = lam[i]
#     	peakA = Re_eps[i]
    	# print(lam[i])

for i in range(0,len(Re_eps)-1):
    if ((Re_eps[i] < Re_eps[i-1]) & (Re_eps[i] < Re_eps[i+1]) & (lam[i]<10)):
    	peak  = lam[i]
    	peakA = Re_eps[i]
###############################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r'$\rm RE(\epsilon(\omega))$', fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0,10)
# plt.ylim(-5,5)

plt.plot(lam, Re_eps, label  = r'$\rm RE(\epsilon(\omega)_{xy})$', color = "black", linewidth = 4)
plt.plot(lam, Re_epsz, label = r'$\rm RE(\epsilon(\omega)_{z})$', color = "blue", linewidth = 4)

plt.scatter(peak, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ \epsilon_{-}(\lambda) = %2.2f, \ \lambda = %2.2f (\mu m)$" %(peakA, peak))		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22', )
plt.axvspan(5, 9, color='silver', alpha=0.5)
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =lam_res*1e6, color = 'black')

# plt.savefig("XFAgFilmRefComp.pdf")
plt.savefig("RealPermPlotHighlight.png")
###############################################################################
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r'$\rm IM(\epsilon(\omega))$', fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0,10)
# plt.ylim(0,1)

plt.plot(lam, Im_eps, label  = r'$\rm IM(\epsilon(\omega)_{xy})$', color = "black", linewidth = 4)
plt.plot(lam, Im_epsz, label = r'$\rm IM(\epsilon(\omega)_{z})$', color = "blue", linewidth = 4)

plt.scatter(peak, max(Im_eps), linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ \epsilon_{resonant} = %2.2f $" %peak)		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22')
plt.axvspan(5, 9, color='silver', alpha=0.5)

# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAgFilmRefComp.pdf")
plt.savefig("ImaginaryPermPlotHighlight.png")


# plt.show()

###############################################################################
peak = 1/(peak*1e-4)
fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r'$\rm RE(\epsilon(\omega))$', fontsize = '30')   
plt.xlabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '30')
plt.xlim(400,2000)
# plt.ylim(-5,5)

plt.plot(1/(lam*1e-4), Re_eps, label  = r'$\rm RE(\epsilon(\omega)_{xy})$', color = "black", linewidth = 4)
plt.plot(1/(lam*1e-4), Re_epsz, label = r'$\rm RE(\epsilon(\omega)_{z})$', color = "blue", linewidth = 4)

plt.scatter(peak, peakA, linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ \epsilon_{-}(\nu) = %2.2f, \ \nu = %2.2f (cm^{-1})$" %(peakA, peak))		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22', )
plt.axvspan(1/(5.5*1e-4), 1/(9*1e-4), color='silver', alpha=0.5)
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =lam_res*1e6, color = 'black')

# plt.savefig("XFAgFilmRefComp.pdf")
plt.savefig("RealPermPlotHighlightWN.png")
###############################################################################

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel(r'$\rm IM(\epsilon(\omega))$', fontsize = '30')   
plt.xlabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '30')
plt.xlim(400,2000)
# plt.ylim(0,1)

plt.plot(1/(lam*1e-4), Im_eps, label  = r'$\rm IM(\epsilon(\omega)_{xy})$', color = "black", linewidth = 4)
plt.plot(1/(lam*1e-4), Im_epsz, label = r'$\rm IM(\epsilon(\omega)_{z})$', color = "blue", linewidth = 4)

plt.scatter(peak, max(Im_eps), linewidth = 3, s=55,edgecolors = 'black', c='red', zorder = 25, label = r"$ \epsilon_{resonant} = %2.2f $" %peak)		

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22')
plt.axvspan(1/(5.5*1e-4), 1/(9*1e-4), color='silver', alpha=0.5)

# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAgFilmRefComp.pdf")
plt.savefig("ImaginaryPermPlotHighlightWN.png")


# plt.show()

###############################################################################