#hi from the outside
from numpy import *
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
Lambda = loadtxt("pbg3.txt", usecols=(0,),  unpack =True)
Pitch = loadtxt("pbg3.txt", usecols=(1,),  unpack =True)

plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
pfont = {'fontname' : 'Times'} 
xlabel(r'Resonant Wavelength $\rm (\mu m)$', pfont, fontsize = '20')
ylabel(r'Pitch $\rm (\mu m)$', pfont, fontsize = '20')

#ylim(0,2)
#xlim(0,34)
#xticks(coord, symm, fontname = 'Times', fontsize = '30')
#title('DField', pfont, fontsize = '30')
plot(Pitch, Lambda)
plt.tight_layout()
plt.show()

