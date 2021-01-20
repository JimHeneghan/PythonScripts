import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *
import matplotlib.patches as mpatches
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                AutoMinorLocator)

thick = [60, 105, 210, 490, 1020, 1990, 6400]
shades = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']
ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']
# ShadesRed  = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray',
 # 'lightgrey', 'gainsboro','darkslategray', 'darkslategrey','lightslategray', 'lightslategrey', 'slategrey', 'slategray']
ShadesRed.insert(4,  'blue')
ShadesRed.insert(9,  'lime')
ShadesRed.insert(14, 'red')

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (18,30), sharex = True)
# fig, ax = plt.subplots( figsize = (9,12))
fig.subplots_adjust(hspace=0)
plt.gcf().subplots_adjust(bottom=0.24, top = 0.99)

# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel('Reflectance', rotation='vertical', fontsize = '70', labelpad = 50)

# plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=0)
# tick_params(direction = 'in', width=4, labelsize=30)
# rc('axes', linewidth=2) 

#plt.tight_layout(pad=5.4, w_pad=5.5, h_pad=5.4)

c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)

########################################################################################
for i in range (0, 20):
	base = 2 + 4*i
	namy = "RTADataBareDevice/2_%.02dumPitchRTA.txt" %base
	lam, A = loadtxt(namy, usecols=(0,1), skiprows= 1, unpack =True)
	print(namy)
	lam = 1/(lam*1e-4)
	ax1.plot(lam, A,  color = ShadesRed[i], linewidth = 3)


plt.setp(ax1.spines.values(), linewidth=4)
ax1.tick_params(direction = 'in', width=5, size = 10, labelsize=30)
ax1.tick_params(which ='minor', direction = 'in', width=3, size = 5)
ax1.xaxis.set_major_locator(MultipleLocator(100))
ax1.xaxis.set_minor_locator(MultipleLocator(50))

ax1.text(.7,.9,'Bare Device',
        horizontalalignment='center',
        transform=ax1.transAxes, fontsize = '40')   
ax1.set_yticks(np.arange(0.5, 1.0, 0.1))

# ax2.legend(loc='upper right', fontsize='12')
ax1.axvline(x =1/(7.26*1e-4), color = 'black')
ax1.set_xlim(1000,1800)
ax1.set_ylim(0.4,1)


########################################################################################

plt.setp(ax2.spines.values(), linewidth=4)
ax2.tick_params(direction = 'in', width=5, size = 10, labelsize=30)
ax2.tick_params(which ='minor', direction = 'in', width=3, size = 5)
ax2.xaxis.set_major_locator(MultipleLocator(100))
ax2.xaxis.set_minor_locator(MultipleLocator(50))
ax2.text(.7,.9,'Uncoupled Device',
        horizontalalignment='center',
        transform=ax2.transAxes, fontsize = '40')   
ax2.set_yticks(np.arange(0.5, 1.0, 0.1))

for i in range (0, 20):
	base = 2 + 4*i
	namy = "RTADataDeadDevice/2_%.02dumPitchRTA.txt" %base
	print(namy)
	lam, A = loadtxt(namy, usecols=(0,1), skiprows= 1, unpack =True)
	lam = 1/(lam*1e-4)
	ax2.plot(lam, A,  color = ShadesRed[i], linewidth = 3)

ax2.axvline(x =1/(7.26*1e-4), color = 'black')
ax2.set_xlim(1000,1800)
ax2.set_ylim(0.4,1)
ax2.legend(loc='upper right', fontsize='12')


# ########################################################################################

plt.setp(ax3.spines.values(), linewidth=4)
ax3.tick_params(direction = 'in', width=5, size = 10, labelsize=30)
ax3.tick_params(which ='minor', direction = 'in', width=3, size = 5)
ax3.tick_params('x', pad = 10)

ax3.xaxis.set_major_locator(MultipleLocator(100))
ax3.xaxis.set_minor_locator(MultipleLocator(50))# ax4.set_ylabel("Absorption", fontsize = '20')   

ax3.set_yticks(np.arange(0.5, 1.0, 0.1))

for i in range (0, 20):
	base = 2 + 4*i
	namy = "RTADataFullDevice/2_%.02dumPitchRTA.txt" %base
	lam, A = loadtxt(namy, usecols=(0,1), skiprows= 1, unpack =True)
	lam = 1/(lam*1e-4)
	ax3.plot(lam, A,  color = ShadesRed[i], linewidth = 3)

ax3.axvline(x =1/(7.26*1e-4), color = 'black')
ax3.set_xlim(1000,1800)
ax3.set_ylim(0.4,1)
# ax4.yaxis.set_label_position("right")
ax3.text(.7,.9,'Coupled Device',
        horizontalalignment='center',
        transform=ax3.transAxes, fontsize = '40')
# ax4.text(1, 0.9, "Coupled", fontsize = '40')   

ax3.set_xlabel(r"$\rm Frequency\ (cm^{-1})$", fontsize = '70')


###############################################################################
patches = []
PitchLeng = np.linspace(2.02, 2.78, 20)
dcc = np.zeros(20, dtype = np.double)
dcc = PitchLeng

for i in range(0, 20):
    temp = mpatches.Patch(facecolor=ShadesRed[i], label = r'$ d_{cc} \rm = %2.2f \ \mu m$' %dcc[i], edgecolor='black')
    patches.append(temp) 
leg = ax3.legend(handles = patches, ncol = 4, loc = 'lower center', frameon = True,fancybox = False, 
fontsize = 30, bbox_to_anchor=(-0.1, -0.75, 1.2, .175),mode="expand", borderaxespad=0.) #bbox_to_anchor=(1.05, 0.5),
leg.get_frame().set_edgecolor('black')

leg.get_frame().set_linewidth(4)
###############################################################################  

plt.savefig("Fig_S3_Ref_ALLines.png")
plt.savefig("Fig_S3_Ref_ALLines.pdf")

# Peak.insert(i, min(RT[0:300]))