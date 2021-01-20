#Lbraries *******************************************************************
from numpy import *
from scipy import *
from pylab import *
import math
#opening a file that will store all the values for our PBG diagram
f = open("pbg3.txt", "w")
val = 0
for i in range (3,4):
    E1 = "SilverCompareLum%d" %i
    E2 = ".txt"
    E = E1 + E2
    finder = E1 + ".txt"
    omega = loadtxt(E, usecols=(0,), skiprows= 1, unpack =True)
    E0 = loadtxt(E, usecols=(1,), skiprows= 1, unpack =True)
    a = (0.5)
    GammaX = 2*math.pi/(a*math.sqrt(3))
    GammaJ = 2*GammaX/math.sqrt(3)
    XJ = GammaX/math.sqrt(3)

    plot(omega, E0)
    show()

#    plot(omega, (log(E0))**2)
#    show()
#    if (i<101):
#        val = i*GammaX/100
#        print "GX"
#        print i
#    if(i>100 and i<201):
#        val = GammaX + (i-100)*GammaJ/100
#        print "GJ"
#        print i
#    if (i>200):
#        print "XJ"
#        print i
#        val = GammaX + GammaJ +(i-200)*XJ/50
    M = len(E0)
    print GammaX + GammaJ + XJ

    print E
    print "\n"
#finding out how many points are in our range
# finding the maximum point in our window
    for j in range (0, M):
	Max = omega[j]
	if (Max > 7e14):
		break


#finding peaks for every point in our range
    for i in range (1, j):
	if(E0[i] > 1e6):
    		if ((E0[i-1] < E0[i])and( E0[i] > E0[i+1])):
			thang1 = str(val)
                        thang2 = str(omega[i])
#recording every peak for each file along with the bloch vector for that file
                        f.write(thang1)
                        f.write("\t")
                        f.write(thang2)
                        f.write("\n")
    val = val + 1


f.close()
