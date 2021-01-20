#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt

time = linspace(1, 100000)
ppw = 30
location = 0.0
cdtds = 1/sqrt(3)
arg = math.pi * ((cdtds * time - location) / ppw - 1.0);
arg = arg * arg;

y = (1 - 2*arg)*np.exp(-arg)

#plot(time, y)
plot(fft(time), fft(y))
show()


