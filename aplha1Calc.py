#hi from the outside
from numpy import *
from scipy import *
from pylab import *
import math

lam = loadtxt("hBN_RT2.txt", usecols=(0,),  unpack =True)
T = loadtxt("hBN_RT2.txt", usecols=(2,),  unpack =True)
R = loadtxt("hBN_RT2.txt", usecols=(1,),  unpack =True)

d = 83e-9
alpha_1 = (-log(T/(1-R)))/d

xlim(5,10)
plot(lam, alpha_1)
show()
