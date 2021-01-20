import numpy as np
from scipy import interpolate
import pylab as py

freq1 = np.loadtxt("Perovskite_EpsDark.txt", usecols=(0,), skiprows= 1,  unpack =True)
ReEps = np.loadtxt("Perovskite_EpsDark.txt", usecols=(1,), skiprows= 1,  unpack =True)

freq2 = np.loadtxt("Perovskite_EpsDark.txt", usecols=(2,), skiprows= 1,  unpack =True)
ImEps = np.loadtxt("Perovskite_EpsDark.txt", usecols=(3,), skiprows= 1,  unpack =True)

#ImEps = -1*ImEps
func = interpolate.interp1d(freq1, ReEps)

print len(freq1)
print len(freq2)
print len(func(freq2))

#py.plot(lam1, ReEps, 'ro')
py.plot(freq2, ImEps, 'go')
py.plot(freq1, func(freq2), 'bo')
py.show()


ff = open('Perm.txt', 'w')
for k in range (0, len(freq2)):
        thang1 = str(freq2[k])
        thang2 = str(-1*ImEps[k])
        thang3 = str(func(freq2)[k])
        ff.write(thang1)
        ff.write("\t")
        ff.write(thang2)
        ff.write("\t")
        ff.write(thang3)
        ff.write("\n")
ff.close()
