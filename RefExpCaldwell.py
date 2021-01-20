import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *

#R = np.zeros((6, ))

d =[60, 105, 210, 490, 1990, 6400]

fig, ax = plt.subplots(figsize=(12,9))
plt.tight_layout()

for i in range (0, 6):
    hBNFile1 = "hBN%d" %d[i]
    hBNFile2 = "nmRef.txt"
    hBNFile = hBNFile1 + hBNFile2
    k, R_exp = loadtxt(hBNFile, usecols=(0,1,),  unpack = True)
    ax.plot(k, R_exp, label = "hBN thickness %g nm" %(d[i]), zorder=(len(d) -i))
    


rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)
ylabel("R", fontsize = '30')
xlabel(r'$\lambda (\mu m)$', fontsize = '30')


#plt.plot(k, Rlow, label = r"fitted gamma is %g $ \rm cm^{-1}$" %gammaLow)
ax.legend(loc='upper left', fontsize='10')
#ax.legend(bbox_to_anchor=(0.1, 1), fancybox=True, shadow=True, borderaxespad=30)
plt.tight_layout()
#fig.savefig('samplefigure', bbox_inches='tight')

plt.show()
