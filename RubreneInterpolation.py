import numpy as np
from scipy import interpolate
import pylab as py

lam1 = np.loadtxt("RubreneReEpsReverse.txt", usecols=(0,), skiprows= 1,  unpack =True)
ReEps = np.loadtxt("RubreneReEpsReverse.txt", usecols=(1,), skiprows= 1,  unpack =True)

lam2 = np.loadtxt("RubreneImEpsReverse.txt", usecols=(0,), skiprows= 1,  unpack =True)
ImEps = np.loadtxt("RubreneImEpsReverse.txt", usecols=(1,), skiprows= 1,  unpack =True)


func = interpolate.interp1d((lam2, ImEps)(np.linspace(0.8, 0.23, num=64)))
print len(lam1)
print len(lam2)
#lam3 = lam2[2:len(lam2)]
#print len(func(lam2))

py.plot(lam1, ReEps, 'ro')
py.plot(lam2, ImEps, 'go')
py.plot(lam2, func(lam1), 'bo')
py.show()


