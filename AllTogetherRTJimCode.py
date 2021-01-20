import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *

thick = [60, 105, 210, 490, 1020, 1990, 6400]
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (9,12), sharex = True)
# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel('A', rotation='vertical', fontsize = '35', labelpad = 40)

fig.subplots_adjust(hspace=0)
# plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=0)
plt.setp(ax1.spines.values(), linewidth=2)
tick_params(direction = 'in', width=2, labelsize=20)
rc('axes', linewidth=2) 

#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)
plt.setp(ax1.spines.values(), linewidth=2)
ax1.tick_params(direction = 'in', width=2, labelsize=20) 
# ax1.set_ylabel("Absorption", fontsize = '20') 
ax1.set_yticks(np.arange(0.1, 0.6, 0.1))
lam, R, T, A = np.loadtxt("hBN_VacRTA.txt", usecols=(0,1,2,3), skiprows= 1, unpack =True )
lam = 1/(lam*1e-4)

c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)


ax1.set_xlim(1000,2000)
ax1.set_ylim(0,0.42)
minRT = 1
mindex = 0


ax1.plot(lam, A,  color = "black", linewidth = 3)

# ax1.legend(loc='center right', fontsize='18')
ax1.axvline(x =1/(7.26*1e-4), color = 'black')

########################################################################################


lam, A10 = loadtxt("2_10umPitchRTA_Bare.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam, A46 = loadtxt("2_46umPitchRTA_Bare.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam, A58 = loadtxt("2_58umPitchRTA_Bare.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam = 1/(lam*1e-4)

ax2.plot(lam, A10,  color = "blue", linewidth = 3, label = r"$\rm d_{cc} = 2.10 \ \mu m$")
ax2.plot(lam, A46,  color = "black", linewidth = 3, label = r"$\rm d_{cc} = 2.46 \ \mu m$")
ax2.plot(lam, A58,  color = "red", linewidth = 3, label = r"$\rm d_{cc} = 2.58 \ \mu m$")

plt.setp(ax2.spines.values(), linewidth=2)
ax2.tick_params(direction = 'in', width=2, labelsize=20)
# ax2.set_ylabel("Absorption", fontsize = '20')   
ax2.set_yticks(np.arange(0.1, 0.6, 0.1))

ax2.legend(loc='upper right', fontsize='12')
ax2.axvline(x =1/(7.26*1e-4), color = 'black')
ax2.set_xlim(1000,2000)
ax2.set_ylim(0,0.42)


########################################################################################

plt.setp(ax3.spines.values(), linewidth=2)
ax3.tick_params(direction = 'in', width=2, labelsize=20)
# ax3.set_ylabel("Absorption", fontsize = '20')   
ax3.set_yticks(np.arange(0.1, 0.6, 0.1))

# ax3.legend(loc='upper right', fontsize='10')
lam, A10 = loadtxt("2_10umPitchRTA_Dead.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam, A46 = loadtxt("2_46umPitchRTA_Dead.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam, A58 = loadtxt("2_58umPitchRTA_Dead.txt", usecols=(0,3), skiprows= 1, unpack =True)

lam = 1/(lam*1e-4)

ax3.plot(lam, A10,  color = "blue", linewidth = 3)
ax3.plot(lam, A46,  color = "black", linewidth = 3)
ax3.plot(lam, A58,  color = "red", linewidth = 3)
# for i in range(0,len(A)-1):
#     if ((A[i] > A[i-1]) & (A[i] > A[i+1])):
#     	peak  = lam[i]
#     	peakA = A[i]
#     	# print(lam[i])
# for i in range(0,len(A)-1):
#     if ((A[i] > A[i-1]) & (A[i] > A[i+1]) & (lam[i] < 7.5)):
#     	peak2  = lam[i]
#     	peakA2 = A[i]



ax3.axvline(x =1/(7.26*1e-4), color = 'black')
ax3.set_xlim(1000,2000)
ax3.set_ylim(0,0.42)
# ax3.scatter(peak, peakA,   linewidth = 3, s=55,edgecolors = 'black', c='red',  zorder = 25, label = r"$A_{ Plasmon   \ peak = %2.2f \ \mu m}$" %peak)		
# ax3.scatter(peak2, peakA2, linewidth = 3, s=55,edgecolors = 'black', c='blue', zorder = 25, label = r"$A_{ Polariton \ peak = %2.2f \ \mu m}$" %peak2)		
# ax3.axvline(x =peak,  color = 'red')
# ax3.axvline(x =peak2, color = 'blue')

########################################################################################

plt.setp(ax4.spines.values(), linewidth=2)
ax4.tick_params(direction = 'in', width=2, labelsize=20)
# ax4.set_ylabel("Absorption", fontsize = '20')   
ax4.set_yticks(np.arange(0.1, 0.6, 0.1))

# ax4.legend(loc='upper right', fontsize='10')
lam, A10 = loadtxt("2_10umPitchRTA_Full.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam, A46 = loadtxt("2_46umPitchRTA_Full.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam, A58 = loadtxt("2_58umPitchRTA_Full.txt", usecols=(0,3), skiprows= 1, unpack =True)
lam = 1/(lam*1e-4)

ax4.plot(lam, A10,  color = "blue", linewidth = 3)
ax4.plot(lam, A46,  color = "black", linewidth = 3)
ax4.plot(lam, A58,  color = "red", linewidth = 3)

ax4.axvline(x =1/(7.26*1e-4), color = 'black')
ax4.set_xlim(1000,2000)
ax4.set_ylim(0,0.42)
ax4.set_xlabel(r"$\rm Frequency\ (cm^{-1})$", fontsize = '35')

plt.savefig("Fig3_4Pane_WN.png")
plt.savefig("Fig3_4Pane_WN.pdf")

# Peak.insert(i, min(RT[0:300]))