import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
rain = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
Nx = 119
Ny = 1100
fig, ax = plt.subplots(1, figsize = (15, 9),constrained_layout=True)

Field = "XSec2_38um/EyHeatMap6.80.txt"
Full = np.zeros((Ny, Nx), dtype = np.double)

print(Field)

Y = np.linspace(0,  Ny, Ny)
X = np.linspace(-1.19,  1.19, Nx)


Ey = np.loadtxt(Field, usecols=(0), skiprows= 1, unpack =True )


for i in range (0, Ny):
	Full[i] = Ey[i*Nx: i*Nx + Nx]

ax.plot(X, Full[614], linewidth=2, color = "black", label = r"$ \rm 6.5 \ \mu m$")

peakx = []
peakE = []

for i in range(0,len(X)-1):
    if ((Full[614,i] > Full[614,i-1]) & (Full[614,i] > Full[614,i+1])):
        # print(Efft[i], 1.0/k[i], "um", 2*math.pi*k[i]*1e4, "cm-1")
        peakx.append(X[i])
        peakE.append(Full[614,i])

ax.scatter(peakx, peakE, linewidth = 2, s=70,edgecolors = 'black', zorder = 25, c='gold')

for i in range (0, len(peakx)):
	ax.axvline(x =  peakx[i], linestyle = "dashed", color = 'black')


ax.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '20')
ax.set_ylabel(r"$ \rm Intensity $", fontsize = '20')
ax.tick_params(direction = 'in', width=2, labelsize=20, bottom = False)
peakx = np.around(peakx, 2)
ax.set_xticks(peakx)
ax.set_xticklabels(peakx)
plt.setp(ax.spines.values(), linewidth=2)



plt.savefig("6.8umEyExcitationLineout2_38_SurfaceDepthDepth.png") 
plt.clf()