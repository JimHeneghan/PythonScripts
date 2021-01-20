from scipy import *
from pylab import *


# Load Complex Dielectric Function Data
freq, Rexx, Rezz = loadtxt("hBNgetindexConstZ.txt", usecols=(0,1,5), skiprows = 1, unpack = True)

plot(freq, Rexx)
plot(freq, Rezz)
show()

plot(freq*1e10/3e8, Rexx)
plot(freq*1e10/3e8, Rezz)
show()
