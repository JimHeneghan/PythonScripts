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
fig, (ax1, ax3, ax2) = plt.subplots(3, figsize = (9,15))

plt.setp(ax1.spines.values(), linewidth=2)
ax1.tick_params(direction = 'in', width=2, labelsize=10)
ax1.tick_params(axis = 'x', direction = 'in', width=2, labelsize=0)
ax1.set_xlim(1200,1700)
plt.rc('axes', linewidth=2) 

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
    ax1.plot(k_data, func(k_data, *popt), linewidth = 2, color = shades[z], label = r"$d\rm_{hBN}  \ = \ %d \ nm$" %thick[z])# label = r"$\rm \gamma_{fit} = %.2f \ J = %.2f $ " %(popt[0], popt[1]))# "\n" r"J = %f" %tuple(popt))
    print"thickness = %d \t gamma = %0.2f \t J = %0.2f " %(thick[z], popt[0], popt[1])
    ax1.plot(k_data, R_exp, 'o',  markersize=3, markeredgecolor = "black", markerfacecolor = shades[z])#, label = "Digitized Data")#  markersize=3, markeredgecolor = "black", markerface

        

    #    legend(loc='lower center', fontsize='30')
ax1.set_ylabel(r"$R$", fontsize = '35')
ax1.set_xlabel(r'$\rm \nu \ (cm^{-1})$', fontsize = '35')

# ax1.set_xlabel(r'$\rm Wavenumber \ (cm^{-1})$', fontsize = '35')
ax1.legend(loc='center right', fontsize='15')
ax1.axvline(x = 1370, linestyle = "dashed", color = 'black', linewidth = 2)
ax1.axvline(x = 1610, linestyle = "dashed", color = 'black', linewidth = 2)

ax3.tick_params(direction = 'in', width=2, labelsize=10, size = 4)
ax3.tick_params(which ='minor', direction = 'in', width=2, size = 2)
ax3.set_ylabel(r'$\gamma_{x,y} \ \rm  (rad \ s^{-1})$', fontsize = '35')
ax3.set_xlabel(r'$ d\rm_{hBN}$', fontsize = '35')

# thick = [60e-9, 105e-9, 210e-9, 490e-9, 1020e-9, 1990e-9, 6400e-9]

def func(d, m, c):
    return m*d + c

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
 
# num1 = yfit(log(83))
# num2 = yfit(1000)

# num1 = 2*math.pi*c0*(exp(num1)*100/1e12)
# print("yfit for 83 nm is %5.3f" %num1)
print(r"$\rm log(\gamma) = %5.3f log(d) \ + \ log(%5.3f)$"  %tuple(popt))


ax3.loglog(thick, 2*math.pi*c0*y, "ro", markersize=10, markeredgecolor = "black", markerfacecolor = 'red', label = r"$\mathrm{Calculated \ \mathit{\gamma}_{x,y}}$")# shades[z])#, label = r"$\rm Calculated \ \gamma: \ d = %d $" %thick[z], zorder = 5)
ax3.loglog(x, 2*math.pi*c0*yfit(x), color = "black", linewidth=3, label = r"$\mathrm{Power-law \ fit \ to} \ \gamma_{x,y} \mathrm{ \ vs \ } d\rm_{hBN} \mathrm{\ data}$", zorder = 1)
ax3.scatter(80, 2*math.pi*c0*(exp(3.484986312)*100), s = 100, facecolors = "none", edgecolor = "limegreen",linewidth=3.0, zorder = 6, label = r"$\gamma_{x,y} \mathrm{ \ (80 \ nm) = %5.3f \times 10^{12} \ (rad \ s^{-1})} $" %(2*math.pi*c0*(exp(3.484986312)*100/1e12)))

ax3.legend(fancybox = True, loc='lower center', fontsize='15', framealpha=1)
# plt.ylim([1, 2*math.pi/10**2])
ax3.set_xlim([(10**2)/2, 10**4])
ax3.axvline(x =80, color = 'black')
ax3.axhline(y = 2*math.pi*c0*(exp(3.484986312)*100), color = 'black')
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
ax2.set_ylabel("R, T, A", fontsize = '35')   
ax2.set_xlabel(r"$\rm Wavenumber \ (cm^{-1})$", fontsize = '35')

R = np.loadtxt("Ref.txt", usecols=(0), skiprows= 1, unpack =True )
I = np.loadtxt("Inc.txt", usecols=(0), skiprows= 1, unpack =True )
T = np.loadtxt("Trans.txt", usecols=(0), skiprows= 1, unpack =True)

c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)

Ifft = np.fft.fft(I, len(R))
Rfft = np.fft.fft(R, len(R))
Tfft = np.fft.fft(T, len(R))

Ref =  (abs(Rfft)/abs(Ifft))**2
Trans = (abs(Tfft)/abs(Ifft))**2
RT = (abs(Tfft)/abs(Ifft))**2 + (abs(Rfft)/abs(Ifft))**2
fs = 1/(dt*len(R))
f = fs*np.arange(0,len(R))
Lambda = c0/f

ax2.set_xlim(1200,1700)
ax2.set_ylim(0,1)
print len(RT)
print min(RT[0:1000])
wn = (1/(Lambda))/100

ax2.plot((k0*1e-2), Rm, label=r'$\rm R_{TM}$', color='darkred', linewidth = 4)
ax2.plot((k0*1e-2), Tm, label=r'$\rm T_{TM}$', color='blue', linewidth = 4)
ax2.plot((k0*1e-2), (1-(Rm + Tm)),  label=r'$\rm A_{TM}$', color='green', linewidth = 4)

ax2.plot(wn,Ref, label = r'$\rm R_{FDTD}$', color = "red", linewidth = 2)
ax2.plot(wn,Trans, label = r'$\rm T_{FDTD}$', color = "black", linewidth = 2)
ax2.plot(wn,(1-RT), label = r'$\rm A_{FDTD}$', color = "limegreen", linewidth = 2)

ax2.legend(loc='center right', fontsize='18')
ax2.axvline(x = 1370, color = 'black', linewidth = 2)

plt.tight_layout()
plt.savefig("AllTogether_JimCode_Working_Nov11.png")
plt.savefig("AllTogether_JimCode_Working_Nov11.pdf")
# plt.show()
