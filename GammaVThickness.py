import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.optimize import curve_fit
from pylab import *

thick = [60.0, 105.0, 210.0, 490.0, 1020.0, 1990.0, 6400.0 ]
gamma = [40.047, 27.2056, 9.622, 17.4781, 8.7441, 9.0982, 1.3777]
#d = linspace(60e-9, 6400e-9, 6341)
fig, ax = plt.subplots(figsize=(12,9))

plt.tight_layout()
coords = [4, log(83), 5, 6, 7, 8, 9]
labels = ["4", "83 nm", "5", "6", "7", "8", "9"]
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
xlabel("hBN Layer Thickness (log(nm))", fontsize = '30')
xticks(coords, labels,  fontsize = '15')
ylabel(r'$\rm \gamma (log(cm^{-1}))$', fontsize = '30')

#ax.legend(loc='lower left', fontsize='10')
plt.tight_layout()
#d1 = [60, 105, 210, 490, 1990, 6400]

# base =  linspace(2, 2.1, 10)
# Amp = linspace(4.89e-12, 5e-12, 10)
# c = linspace(1100, 1120, 10)
# y = np.zeros(len(thick))
# chi2 = 300000
# for j in range (0, len(base)):
#     for k in range (0, len(Amp)):
#     	for l in range (0, len(c)):
# 	        for i in range (0, len(thick)):
# 	        	y[i] = (Amp[k]*((thick[i]*1e-9)**(base[j]))+c[l])/100
# 	            #ax.plot(thick, y, label = "base is %g" %base[j]) 
# 	        chi1 = sum((gamma - y)**2)
# 	        if (chi1 < chi2):
# 	            ylow = y
# 	            baselow = j
# 	            Amplow = k
# 	            q = 1e-9
# 	            ax.plot(np.asarray(thick)*q, np.asarray(y)*100, label = "base is %g" %base[j]) 
# print chi1
# print base[baselow]
#print Amp[Amplow]
def func(d, m, c):
	return m*d + c

popt, pcov = curve_fit(func, log(np.asarray(thick)) , log(np.asarray(gamma)))

print thick
print "m is %f, c is %f" %tuple(popt)
gam = (-0.588507)*log(83) + 6.065841
print gam 

#ax.plot(d*1e9, y/100)#, label = "base is %g" %base[baselow])    
plt.tight_layout()
plt.setp(ax.spines.values(), linewidth=2)  
plt.scatter(log(83), 3.46532237043, s = 100, facecolors = "none", edgecolor = "red",linewidth=3.0, zorder = 5)
plt.scatter(log(thick), log(gamma), s = 100, edgecolor = "black", facecolors = "blue", label = "Calculated Data")
#ax.plot(log(thick), log(gamma), 'o', label = r" $\rm Thickness \ v \ \gamma_{Fit}$")
axvline(x =log(83), color = 'black')
axhline(y = 3.46532237043, color = 'black')
plt.plot(log(np.asarray(thick)), func(log(np.asarray(thick)), *popt), label = "fitted Data", color = "black")

ax.legend(fancybox = True, loc='upper center', fontsize='25', framealpha=1)
#ylim(0, 100)
plt.savefig("BlackEdgeLog_Log_TVgLinFit83Line.png")
plt.savefig("BlackLog_Log_TVgLinFit83Line.pdf")
plt.show()
