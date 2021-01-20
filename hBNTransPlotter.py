#hi from the outside
from numpy import *
from scipy import *
from pylab import *
import math

lam = loadtxt("hBN_RTConstZ32.txt", usecols=(0,),  unpack =True)
R =   loadtxt("hBN_RTConstZ32.txt", usecols=(1,),  unpack =True)
T =   loadtxt("hBN_RTConstZ32.txt", usecols=(2,),  unpack =True)

xlim(5,10)
plot(lam, T, label = "T", color = "blue")
plot(lam, R, label = "R", color = "red")
plot(lam, R+T, label = "R+T", color = "Green")
legend(loc= 'center right', fontsize='20')
show()
