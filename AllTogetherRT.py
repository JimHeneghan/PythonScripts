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
ax0.set_ylabel('Absorption', rotation='vertical', fontsize = '35', labelpad = 40)

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
Lambda, R, T, RT = loadtxt("NewGamma7.5psRT5_22.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)

ax1.set_xlim(5,9)
ax1.set_ylim(0,0.6)
minRT = 1
mindex = 0
# for i in range (0, len(RT)):
# 	if (RT[i] < minRT):
# 		minRT = RT
# 		mindex = i
print np.argmin(RT)
print Lambda[151]
ax1.plot(Lambda,(1-RT),  color = "black", linewidth = 3)

# ax1.legend(loc='center right', fontsize='18')
ax1.axvline(x =7.245885769603097, color = 'black')


E = "2.281PitchRT.txt"
print E
Lambda, R,T,RT = loadtxt(E, usecols=(0,1,2,3), skiprows= 0, unpack =True)
print len(RT)

ax2.plot(Lambda, (1-RT), linewidth = 3, color = "black")
plt.setp(ax2.spines.values(), linewidth=2)
ax2.tick_params(direction = 'in', width=2, labelsize=20)
# ax2.set_ylabel("Absorption", fontsize = '20')   
ax2.set_yticks(np.arange(0.1, 0.6, 0.1))

# ax2.legend(loc='upper left', fontsize='20')
ax2.axvline(x =7.245885769603097, color = 'black')
ax2.set_xlim(5,9)
ax2.set_ylim(0,0.6)

plt.setp(ax3.spines.values(), linewidth=2)
ax3.tick_params(direction = 'in', width=2, labelsize=20)
# ax3.set_ylabel("Absorption", fontsize = '20')   
ax3.set_yticks(np.arange(0.1, 0.6, 0.1))

# ax3.legend(loc='upper right', fontsize='10')
Lambda, R, T, RT = loadtxt("hBN_Ag_Si2.31umRT.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)

ax3.plot(Lambda,(1-RT), label = "R+T", color = "black", linewidth = 3)
ax3.axvline(x =7.245885769603097, color = 'black')
ax3.set_xlim(5,9)
ax3.set_ylim(0,0.6)


plt.setp(ax4.spines.values(), linewidth=2)
ax4.tick_params(direction = 'in', width=2, labelsize=20)
# ax4.set_ylabel("Absorption", fontsize = '20')   
ax4.set_yticks(np.arange(0.1, 0.6, 0.1))

# ax4.legend(loc='upper right', fontsize='10')
Lambda, R, T, RT = loadtxt("DeadhBN_Ag_Si2.3umRT.txt", usecols=(0,1,2,3), skiprows= 0, unpack =True)

ax4.plot(Lambda,(1-RT), label = "R+T", color = "black", linewidth = 3)

ax4.axvline(x =7.245885769603097, color = 'black')
ax4.set_xlim(5,9)
ax4.set_ylim(0,0.6)
ax4.set_xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '35')

plt.savefig("AllTogether_RT_June9_5.png")
plt.savefig("AllTogether_RT_June9_5.pdf")

# Peak.insert(i, min(RT[0:300]))