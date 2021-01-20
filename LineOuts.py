import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(1, figsize = (7, 8), constrained_layout=True)
ax = fig.add_subplot(111)
# cmatch = ["c", "olive", "purple", "goldenrod", "orangered", "steelblue"]
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
freq = ['6.1', '6.3', '6.4', '6.5', '6.9', '7.0']
for z in range(0, len(freq)):
	print z
	Field = "%sPlane.txt" %freq[z]
	# z = z
	Full = np.zeros((221, 129), dtype = np.double)
	X = np.linspace(0,  2.281, 129)
	Y = np.linspace(0,  4.01, 221)
	
	for i in range (0, 221):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 357, unpack =True )
		Full[i] = E 


	ax.plot(X[8:121] - 1.15, Full[110][8:121] + 1.8*(z), linewidth=2, color = cmatch[len(cmatch)-z - 1], label = r"$ \rm %s \ \mu m$" %freq[z])

	#plt.savefig("EFieldXSec43THz.pdf")
ax.legend(loc = 'upper right',  ncol = 1, fancybox= True, shadow = True, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
# ax.set_ylim(0, 14)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(left = False, bottom = False)
ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax.set_ylabel(r"$\rm \|E\|  \ (V m^{-1})$", fontsize = '35')
plt.savefig("EFieldXSec_Line_AbsE_Line1_WLegend_June10.png")
plt.savefig("EFieldXSec_Line_AbsE_Line1_WLegend_June10.pdf")
# plt.show()

