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
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
freq = ['6.1', '6.3', '6.5', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
for z in range(0, 7):
	print z
	Field = "%s_um.txt" %freq[z]
	# z = z
	x, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )
	x = x - max(x)/2
	# if (z == 2):
	# 	E = E + 0.15

	ax.plot(x*1e6, E + 0.3*(z), linewidth=2, color = shades[len(shades)-z - 1], label = r"$ \rm %s \ \mu m$" %freq[z])

	#plt.savefig("EFieldXSec43THz.pdf")
ax.legend(loc = 'upper left',  ncol = 1, fancybox= True, framealpha = 0.5, shadow = False, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
# ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
# ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax.set_xlim(-1.15, 1.15)
plt.setp(ax.spines.values(), linewidth=1)
plt.tick_params(left = False, bottom = False,labelsize = 'large')
ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. Units)$", fontsize = '35')
plt.savefig("AllLineOutsHalfNov19Centered.png")
plt.savefig("AllLineOutsHalfNov19Centered.pdf")

# fig = plt.figure(1, figsize = (7, 8), constrained_layout=True)
# ax = fig.add_subplot(111)
# for z in range(8, len(freq)):
# 	print z
# 	Field = "%s_um.txt" %freq[z]
# 	# z = z
# 	x, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )

# 	if (z == 12):
# 		E = E + 0.3

# 	ax.plot(x*1e6, E + 0.3*(z), linewidth=2, color = shades[len(shades)-z - 1], label = r"$ \rm %s \ \mu m$" %freq[z])

# 	#plt.savefig("EFieldXSec43THz.pdf")
# ax.legend(loc = 'upper left',  ncol = 1, fancybox= True, shadow = True, fontsize = 12) #bbox_to_anchor=(1.05, 0.5),
# ax.axvline(x = -0.68, linestyle = "dashed", color = 'black')
# ax.axvline(x =  0.68, linestyle = "dashed", color = 'black')
# ax.set_xlim(0.3, 4.1)
# plt.setp(ax.spines.values(), linewidth=1)
# plt.tick_params(left = False, bottom = False)
# ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
# ax.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. Units)$", fontsize = '35')
# plt.savefig("AllLineOutsHalfNov19Centered.png")
# plt.savefig("AllLineOutsHalfNov19Centered.pdf")
# # plt.show()

