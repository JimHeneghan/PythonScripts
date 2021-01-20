import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
freq = ["35Hz.txt","39Hz.txt", "DeadhBN/35Hz.txt","DeadhBN/39Hz.txt", "44Hz.txt", "47Hz.txt", "DeadhBN/44Hz.txt", "DeadhBN/47Hz.txt"]
many = np.zeros((10, 401, 231), dtype = np.double)
j = 0
z = 0
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}
fig, axs = plt.subplots(2, 4, figsize = (10, 8),constrained_layout=True)
for ax in axs.flat:
	Field = freq[z]
	print(z)
	# if (z == 4)or(z == 5)or(z == 10)or(z == 11):
	# 	Full = np.zeros((401, 253), dtype = np.double)
	# 	X = np.linspace(0,  2.53, 253)
	# 	Y = np.linspace(0,  4.01, 401)
	# 	ax.set_aspect(401/253)
	# 	for i in range (0, 400):
	# 		E = np.loadtxt(Field, usecols=(i,), skiprows= 697, unpack =True )
	# 		Full[i] = E 
	# else:
	Full = np.zeros((401, 231), dtype = np.double)
	X = np.linspace(0,  2.31, 231)
	Y = np.linspace(0,  4.01, 401)
	ax.set_aspect(401/231)
	for i in range (0, 400):
		E = np.loadtxt(Field, usecols=(i,), skiprows= 639, unpack =True )
		Full[i] = E 
	print(freq[z])
	print(max(E))
	norm = mpl.colors.Normalize(vmin=0, vmax=3.281958194909)

	im = ax.pcolormesh(X, Y, Full, norm = norm, **pc_kwargs)
	
	
	if (z==0):
		ax.set_title(r"$\rm \lambda_{Ex} = 8.57 \ (\mu m) $" "\n" r"$\rm hBN \ Layer$", fontsize = '17')
	elif (z==1):
		ax.set_title(r"$\rm \lambda_{Ex} = 7.69 \ (\mu m) $" "\n" r"$\rm hBN \ Layer$", fontsize = '17')
	elif (z==2):
		ax.set_title(r"$\rm \lambda_{Ex} = 8.57 \ (\mu m) $" "\n" r"$\rm Const. \ \epsilon \ Layer$", fontsize = '17')
	elif (z==3):
		ax.set_title(r"$\rm \lambda_{Ex} = 7.69 \ (\mu m) $" "\n" r"$\rm Const. \ \epsilon \ Layer$", fontsize = '17')
	elif (z==4):
		ax.set_title(r"$\rm \lambda_{Ex} = 6.82 \ (\mu m) $" "\n" r"$\rm hBN \ Layer$", fontsize = '17')
	elif (z==5):
		ax.set_title(r"$\rm \lambda_{Ex} = 6.38 \ (\mu m) $" "\n" r"$\rm hBN \ Layer$", fontsize = '17')
	elif (z==6):
		ax.set_title(r"$\rm \lambda_{Ex} = 6.82 \ (\mu m) $" "\n" r"$\rm Const. \ \epsilon \ Layer$", fontsize = '17')
	elif (z==7):
		ax.set_title(r"$\rm \lambda_{Ex} = 6.38 \ (\mu m) $" "\n" r"$\rm Const. \ \epsilon \ Layer$", fontsize = '17')
		

	ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
	ax.set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '20')

	z = z + 1

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)
	

cbar = fig.colorbar(im, ax=axs)
cbar.set_label(label = r"$\rm \|E\| \ Field \ (V m^{-1})$", size = '20')

plt.savefig("DeadActiveSellection.pdf")
plt.savefig("DeadActiveSellection.png")
# plt.show()
