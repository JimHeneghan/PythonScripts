import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.optimize import curve_fit
from pylab import *

shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
thick = [60.0, 105.0, 210.0, 490.0, 1020.0, 1990.0, 6400.0 ]
gamma = [40.047, 27.2056, 9.622, 17.4781, 8.7441, 9.0982, 1.3777]
fig, ax = plt.subplots(figsize=(15,9))
plt.tight_layout()
rc('axes', linewidth=2)
ax.tick_params(direction = 'in', width=2, labelsize=20, size = 8)
ax.tick_params(which ='minor', direction = 'in', width=2, size = 4)
xlabel("hBN Layer Thickness (nm)", fontsize = '30')
ylabel(r'$\rm \gamma \ (rad \ s^{-1})$', fontsize = '30')
plt.tight_layout()
plt.rcParams['xtick.direction'] = 'in'
#plt.rcParams['xtick.minor.size'] = 30
#plt.rcParams['xtick.minor.width'] = 3

def func(d, m, c):
	return m*d+ c

popt, pcov = curve_fit(func, log(np.asarray(thick)) , log(np.asarray(gamma)))

x = np.asarray(thick)
y = np.asarray(gamma) 
y = y*100
logx = np.log(x)
logy = np.log(y)
coeffs = np.polyfit(logx,logy,deg=1)
poly = np.poly1d(coeffs)

yfit = lambda x: np.exp(poly(np.log(x)))
 
plt.setp(ax.spines.values(), linewidth=2)  
#ax.spines['bottom'].set_position('0')
# for z in range(0, 7):
plt.loglog(thick, 2*math.pi/y, "ro", markersize=20, markeredgecolor = "black", markerfacecolor = 'red', label = r"$\rm Calculated \ \gamma$", zorder = 5)

plt.loglog(x, 2*math.pi/yfit(x), label = "Fitted Data", color = "black", linewidth=3)
plt.scatter(83, 2*math.pi/(exp(3.46532237043)*100), s = 100, facecolors = "none", edgecolor = "limegreen",linewidth=3.0, zorder = 6, label = r"$\gamma \rm = %5.3f  \ (rad \ s^{-1}) $" %exp(3.46532237043))
ax.legend(fancybox = True, loc='upper center', fontsize='12', framealpha=1)
# plt.ylim([1, 2*math.pi/10**2])
plt.xlim([(10**2)/2, 10**4])
axvline(x =83, color = 'black')
axhline(y = 2*math.pi/(exp(3.46532237043)*100), color = 'black')
plt.savefig("LogLogThickVGammaFit_RadssMay28.png")
plt.savefig("LogLogThickVGammaFit_RadssMay28.pdf")
# plt.show()
