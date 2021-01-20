import numpy as np
from scipy import *
from pylab import *
from cmath import *
from numpy import ctypeslib
from ctypes import *
import matplotlib.pyplot as plt
Time  = loadtxt("T2Time.txt", delimiter= ", ", usecols=(0,), skiprows=3 , unpack =True)
EField = loadtxt("T2Time.txt", usecols=(1,), skiprows= 3, unpack =True)
