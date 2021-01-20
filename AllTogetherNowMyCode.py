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
fig = plt.figure(figsize = (6,10))
outer = gridspec.GridSpec(2, 1, figure = fig, height_ratios = [2,1], hspace= 0.2)
fig1 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=outer[0], hspace = 0)
fig2 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=outer[1])

ax1 = fig.add_subplot(fig1[0] )
ax2 = fig.add_subplot(fig1[1], sharex = ax1)
# fig1.subplots_adjust(hspace = 0)
ax3 = fig.add_subplot(fig2[0])

# ax2.subplot_adjust(top = 0)
# asp = (0.83)
# ax1 = plt.subplot(165)
# ax2 = plt.subplot(265)
# ax2 = plt.subplot(365)
# fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
# plt.tight_layout(pad=2.7, w_pad=2.7, h_pad=2.7)
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
    ax1.plot(k_data, func(k_data, *popt), linewidth = 2, color = shades[z], label = r"$\rm d \ = \ %d \ nm$" %thick[z])# label = r"$\rm \gamma_{fit} = %.2f \ J = %.2f $ " %(popt[0], popt[1]))# "\n" r"J = %f" %tuple(popt))
    print"thickness = %d \t gamma = %0.2f \t J = %0.2f " %(thick[z], popt[0], popt[1])
    ax1.plot(k_data, R_exp, 'o',  markersize=3, markeredgecolor = "black", markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface

        

    #    legend(loc='lower center', fontsize='30')
ax1.set_ylabel("R", fontsize = '15')
# ax1.set_xlabel(r'$\rm Wavenumber \ (cm^{-1})$', fontsize = '15')
ax1.legend(loc='upper right', fontsize='8')
ax1.axvline(x = 1370, linestyle = "dashed", color = 'black', linewidth = 2)
ax1.axvline(x = 1610, linestyle = "dashed", color = 'black', linewidth = 2)
# plt.tight_layout()

    #plt.cla()

ax3.tick_params(direction = 'in', width=2, labelsize=10, size = 4)
ax3.tick_params(which ='minor', direction = 'in', width=2, size = 2)
ax3.set_ylabel(r'$\rm \gamma \ (rad \ s^{-1})$', fontsize = '15')
ax3.set_xlabel('hBN Layer Thickness (nm)', fontsize = '15')
ax3.legend(loc='upper right', fontsize='8')
# ax2.axvline(x =1/(7.29*1e-4), color = 'black')

def func(d, m, c):
    return m*d+ c

popt, pcov = curve_fit(func, log(np.asarray(thick)) , log(np.asarray(gamma)))
c0 = 3e8
x = np.asarray(thick)
y = np.asarray(gamma) 
y = y*100
logx = np.log(x)
logy = np.log(y)
coeffs = np.polyfit(logx,logy,deg=1)
poly = np.poly1d(coeffs)

yfit = lambda x: np.exp(poly(np.log(x)))
 
#ax.spines['bottom'].set_position('0')

ax3.loglog(x, 2*math.pi*c0*yfit(x), label = "Fitted Data", color = "black", linewidth=3)
ax3.loglog(thick, 2*math.pi*c0*y, "ro", markersize=10, markeredgecolor = "black", markerfacecolor = 'red', label = r"$\rm Calculated \ \gamma$")# shades[z])#, label = r"$\rm Calculated \ \gamma: \ d = %d $" %thick[z], zorder = 5)

ax3.scatter(83, 2*math.pi*c0*(exp(3.46532237043)*100), s = 100, facecolors = "none", edgecolor = "limegreen",linewidth=3.0, zorder = 6, label = r"$\gamma \rm = %5.3f \times 10^{12} \ (rad \ s^{-1}) $" %(2*math.pi*c0*(exp(3.46532237043)*100/1e12)))
ax3.legend(fancybox = True, loc='lower center', fontsize='8', framealpha=1)
# plt.ylim([1, 2*math.pi/10**2])
ax3.set_xlim([(10**2)/2, 10**4])
ax3.axvline(x =83, color = 'black')
ax3.axhline(y = 2*math.pi*c0*(exp(3.46532237043)*100), color = 'black')
plt.setp(ax3.spines.values(), linewidth=2)

eps_inf = 4.87
s = 1.83
omega_nu = 137200
gamma = 3198.7#/(2*math.pi)#700.01702
d = 83e-9
imp0 = 376.730313
k0 = linspace (100000, 200000, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k0 - k0*k0)

n1 = np.sqrt(eps1)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0

# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*k0*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))
#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax2.spines.values(), linewidth=2)
ax2.tick_params(direction = 'in', width=2, labelsize=10)
# ax2.set_ylabel("R", fontsize = '15')   
ax2.set_xlabel(r"$\rm Wavenumber \ (cm)^{-1}$", fontsize = '15')
# plt.tight_layout()
R = np.loadtxt("Ref.txt", usecols=(0), skiprows= 1, unpack =True )
I = np.loadtxt("Inc.txt", usecols=(0), skiprows= 1, unpack =True )#../StubbyhBN/StubbyVac/Inc/
T = np.loadtxt("Trans.txt", usecols=(0), skiprows= 1, unpack =True )# Lambda = (1/Lambda)*100
#R = loadtxt(E, usecols=(1,), skiprows= 0, unpack =True)
c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)

Ifft = np.fft.fft(I, len(R))
Rfft = np.fft.fft(R, len(R))
Tfft = np.fft.fft(T, len(R))

Ref =  (abs(Rfft)/abs(Ifft))**2
Tran = (abs(Tfft)/abs(Ifft))**2
RT = (abs(Tfft)/abs(Ifft))**2 + (abs(Rfft)/abs(Ifft))**2
fs = 1/(dt*len(R))
f = fs*np.arange(0,len(R))
Lambda = c0/f

ax2.set_xlim(1200,1700)
ax2.set_ylim(0,1)
print len(RT)
print min(RT[0:1000])
wn = (1/(Lambda))/100
ax2.plot(wn,Ref, label = r'$\rm R_{Lumerical:  Correct \ \gamma}$', color = "red", linewidth = 3)
ax2.plot(wn,Tran, label = r'$\rm T_{Lumerical: Correct \ \gamma}$', color = "black", linewidth = 3)
ax2.plot(wn,RT, label = r'$\rm R+T_{Lumerical: Correct \ \gamma}$', color = "limegreen", linewidth = 3)

ax2.plot((k0*1e-2), Rm, label=r'$\rm R_{Macleod:  Correct \ \gamma}$', color='darkred')
ax2.plot((k0*1e-2), Tm,  label=r'$\rm T_{Macleod: Correct \ \gamma}$', color='green')
ax2.plot((k0*1e-2), Rm + Tm,  label=r'$\rm R+T_{Macleod: Correct \ \gamma}$', color='blue')

ax2.legend(loc='center right', fontsize='8')
ax2.axvline(x = 1370, color = 'black', linewidth = 2)

# plt.tight_layout()
plt.savefig("AllTogetherPixJimCodeOption3WN.png")
plt.savefig("AllTogetherPixJimCodeOption3WN.pdf")
plt.show()
