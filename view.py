#hi from the outside
from numpy import *
from scipy import *
from pylab import *
import math
freq = loadtxt("Kpoint23.txt", usecols=(0,),  unpack =True)
E0 = loadtxt("Kpoint23.txt", usecols=(1,),  unpack =True)

plot(freq, E0)
show()
