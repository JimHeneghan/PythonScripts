import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

NFREQs = 500
freq = np.zeros(NFREQs, dtype = np.double)
for i in range(0, NFREQs):
	freq[i] = 3e8/((1e-6) + i*((9e-6)/NFREQs))

print(freq)
f = open("demofile2.txt", "w")

for i in range(0, NFREQs):
	freqwrite = "%d" %freq[i]
	f.write("              <Expression> \n")
	f.write('                <Formula typeid="6" type="Double" value="' + freqwrite)
	# f.write(freq) 
	f.write('"/>"\n')
	f.write("              </Expression>\n")

f.close()