#Lbraries *******************************************************************
import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *

Time = np.linspace(0, 10000, 10000)

arg = (50 - Time)/10
arg2 = arg*arg

pulse = np.exp(-0.5*arg2)

plot(Time, pulse)
show()