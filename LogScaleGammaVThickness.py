import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.optimize import curve_fit
from pylab import *

thick = [60.0, 105.0, 210.0, 490.0, 1020.0, 1990.0, 6400.0 ]
gamma = [40.047, 27.2056, 9.622, 17.4781, 8.7441, 9.0982, 1.3777]
fig, ax = plt.subplots(figsize=(12,9))
plt.tight_layout()
rc('axes', linewidth=2)
ax.tick_params(direction = 'in', width=2, labelsize=20, size = 8)
ax.tick_params(which ='minor', direction = 'in', width=2, size = 4)
xlabel("hBN Layer Thickness (nm)", fontsize = '30')
ylabel(r'$\rm \gamma \ (cm^{-1})$', fontsize = '30')
plt.tight_layout()
plt.rcParams['xtick.direction'] = 'in'
#plt.rcParams['xtick.minor.size'] = 30
#plt.rcParams['xtick.minor.width'] = 3

def func(d, m, c):
	return m*d+ c

popt, pcov = curve_fit(func, log(np.asarray(thick)) , log(np.asarray(gamma)))

x = np.asarray(thick)
y = np.asarray(gamma) 
logx = np.log(x)
logy = np.log(y)
coeffs = np.polyfit(logx,logy,deg=1)
poly = np.poly1d(coeffs)

yfit = lambda x: np.exp(poly(np.log(x)))
 
plt.setp(ax.spines.values(), linewidth=2)  
#ax.spines['bottom'].set_position('0')
plt.loglog(thick, gamma, "ro", markersize=20, markeredgecolor = "black", markerfacecolor = "red", label = "Calculated Data", zorder = 5)

plt.loglog(x, yfit(x), label = "Fitted Data: \n" r"$\rm log(\it{\gamma}) = \it{m}\rmlog(\it{d}) + c$" "\n" r"$\rm fit: \ \it{m} \rm = %5.3f, \ \it{c} \rm = %5.3f$" %tuple(popt), color = "black", linewidth=3)
plt.scatter(83, exp(3.46532237043), s = 100, facecolors = "none", edgecolor = "red",linewidth=3.0, zorder = 6, label = r"$\gamma \rm = %5.3f  \ (cm^{-1}) $" %exp(3.46532237043))
ax.legend(fancybox = True, loc='upper center', fontsize='12', framealpha=1)
plt.ylim([1, 10**2])
plt.xlim([(10**2)/2, 10**4])
axvline(x =83, color = 'black')
axhline(y = exp(3.46532237043), color = 'black')
#plt.savefig("LogLogThickVGammaFitO15_2.png")
#plt.savefig("LogLogThickVGammaFitO15_2.pdf")
plt.show()
