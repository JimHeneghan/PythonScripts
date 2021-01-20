import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
#all units are in m^-1
wp = 1.15136316e16
gamma = 9.79125662e13

#Debye/Drude formulation
eps_inf = 1.0
eps_s   = 343.744
tau     = 7.90279e-15
sigma   = 3.32069e6
d    = 50e-9

imp0 = 376.730313
eps0 = 8.85418782e-12

w = linspace (100e12, 3600e12,  100)
# k0 = k*1e12
c0 = 3e8
dx = 1e-8
dy = 1e-8


# eps1 = 1 + (wp*wp)/(w*(1j*gamma-w))
eps1 = eps_inf + (eps_s-eps_inf)/(1+1j*w*tau) + sigma/(1j*w*eps0)
n1 = np.sqrt(eps1)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*(w/c0)#*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

# B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
# C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
# Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))


Nx = 11
Ny = 11

NFREQs = 500
nref = 1.0
ntra = 1.0

###########################################################################
###########################  Our Ag #######################################
###########################################################################
freq = np.zeros(NFREQs, dtype = np.double)
m  = np.zeros(Nx, dtype = np.int)
n  = np.zeros(Ny, dtype = np.int)

kx = np.zeros(Nx, dtype = np.double)
ky = np.zeros(Ny, dtype = np.double)

Sxr    = np.zeros((Nx, Ny), dtype = np.complex)
Syr    = np.zeros((Nx, Ny), dtype = np.complex)

kzR    = np.zeros((Nx, Ny), dtype = np.complex)
kzT    = np.zeros((Nx, Ny), dtype = np.complex)

Esr    = np.zeros((Nx, Ny), dtype = np.complex)

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REFOur   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRAOur   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("OurAg/freq.txt",  usecols=(0), skiprows= 0, unpack =True)

ExR    = np.loadtxt("OurAg/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("OurAg/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("OurAg/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("OurAg/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("OurAg/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("OurAg/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("OurAg/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("OurAg/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("OurAg/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("OurAg/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("OurAg/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("OurAg/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("OurAg/ExInc.txt", usecols=(0), skiprows= 0, unpack =True)
ExIncI = np.loadtxt("OurAg/ExInc.txt", usecols=(1), skiprows= 0, unpack =True)

EyIncR = np.loadtxt("OurAg/EyInc.txt", usecols=(0), skiprows= 0, unpack =True)
EyIncI = np.loadtxt("OurAg/EyInc.txt", usecols=(1), skiprows= 0, unpack =True)

EzIncR = np.loadtxt("OurAg/EzInc.txt", usecols=(0), skiprows= 0, unpack =True)
EzIncI = np.loadtxt("OurAg/EzInc.txt", usecols=(1), skiprows= 0 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

ExRef = np.reshape(ExRef, (NFREQs, Nx, Ny), order='C')
EyRef = np.reshape(EyRef, (NFREQs, Nx, Ny), order='C')
EzRef = np.reshape(EzRef, (NFREQs, Nx, Ny), order='C')

ExTra = np.reshape(ExTra, (NFREQs, Nx, Ny), order='C')
EyTra = np.reshape(EyTra, (NFREQs, Nx, Ny), order='C')
EzTra = np.reshape(EzTra, (NFREQs, Nx, Ny), order='C')

ExInc = np.reshape(ExInc, (NFREQs, Nx, Ny), order='C')
EyInc = np.reshape(EyInc, (NFREQs, Nx, Ny), order='C')
EzInc = np.reshape(EzInc, (NFREQs, Nx, Ny), order='C')

kx = -2*math.pi*m/(Nx*dx)
ky = -2*math.pi*n/(Ny*dy)
print(len(EzInc))
for ff in range (0, NFREQs):

	Esr = 0
	lam = c0/freq[ff]
	# print(lam)
	k0 = 2*math.pi/lam
	kzinc = k0*nref

	for i in range (0, Nx):
		for j in range (0, Ny):
			kzR[i, j] = cmath.sqrt((k0*nref)**2 - kx[i]**2 - ky[j]**2)
			kzT[i, j] = cmath.sqrt((k0*ntra)**2 - kx[i]**2 - ky[j]**2)
	Esr = np.sqrt(abs(ExInc[ff])**2 + abs(EyInc[ff])**2 + abs(EzInc[ff])**2)#ExInc[ff]*conj(ExInc[ff]) + EyInc[ff]*conj(EyInc[ff]) + EzInc[ff]*conj(EzInc[ff]))
	
	Sxr = ExRef[ff]/Esr
	Syr = EyRef[ff]/Esr
	Szr = EzRef[ff]/Esr

	Sxt = ExTra[ff]/Esr
	Syt = EyTra[ff]/Esr
	Szt = EzTra[ff]/Esr

	# print(Sxr)
	
	Sxrfft = fftshift(fft2(Sxr))/(Nx*Ny)
	Syrfft = fftshift(fft2(Syr))/(Nx*Ny)
	Szrfft = fftshift(fft2(Szr))/(Nx*Ny)

	Sxtfft = fftshift(fft2(Sxt))/(Nx*Ny)
	Sytfft = fftshift(fft2(Syt))/(Nx*Ny)
	Sztfft = fftshift(fft2(Szt))/(Nx*Ny)

	Sref = abs(Sxrfft)**2 + abs(Syrfft)**2 + abs(Szrfft)**2 #conj(Sxrfft)*Sxrfft + conj(Syrfft)*Syrfft + conj(Szrfft)*Szrfft
	Stra = abs(Sxtfft)**2 + abs(Sytfft)**2 + abs(Sztfft)**2 #conj(Sxtfft)*Sxrfft + conj(Sytfft)*Syrfft + conj(Sztfft)*Szrfft

	ref[ff] = Sref*(real(kzR/kzinc))
	tra[ff] = Stra*(real(kzT/kzinc))

for ff in range(0, NFREQs):
	for j in range(0, Ny):
		for i in range(0, Nx):
			REFOur[ff] =  REFOur[ff] + ref[ff, i, j]
			TRAOur[ff] =  TRAOur[ff] + tra[ff, i, j]

###########################################################################
###########################  Paper Ag #####################################
###########################################################################
freq = np.zeros(NFREQs, dtype = np.double)
m  = np.zeros(Nx, dtype = np.int)
n  = np.zeros(Ny, dtype = np.int)

kx = np.zeros(Nx, dtype = np.double)
ky = np.zeros(Ny, dtype = np.double)

Sxr    = np.zeros((Nx, Ny), dtype = np.complex)
Syr    = np.zeros((Nx, Ny), dtype = np.complex)

kzR    = np.zeros((Nx, Ny), dtype = np.complex)
kzT    = np.zeros((Nx, Ny), dtype = np.complex)

Esr    = np.zeros((Nx, Ny), dtype = np.complex)

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REFPaperAg   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRAPaperAg   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("Odd/freq.txt",  usecols=(0), skiprows= 0, unpack =True)

ExR    = np.loadtxt("Odd/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("Odd/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("Odd/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("Odd/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("Odd/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("Odd/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("Odd/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("Odd/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("Odd/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("Odd/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("Odd/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("Odd/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("Odd/ExInc.txt", usecols=(0), skiprows= 0, unpack =True)
ExIncI = np.loadtxt("Odd/ExInc.txt", usecols=(1), skiprows= 0, unpack =True)

EyIncR = np.loadtxt("Odd/EyInc.txt", usecols=(0), skiprows= 0, unpack =True)
EyIncI = np.loadtxt("Odd/EyInc.txt", usecols=(1), skiprows= 0, unpack =True)

EzIncR = np.loadtxt("Odd/EzInc.txt", usecols=(0), skiprows= 0, unpack =True)
EzIncI = np.loadtxt("Odd/EzInc.txt", usecols=(1), skiprows= 0 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

ExRef = np.reshape(ExRef, (NFREQs, Nx, Ny), order='C')
EyRef = np.reshape(EyRef, (NFREQs, Nx, Ny), order='C')
EzRef = np.reshape(EzRef, (NFREQs, Nx, Ny), order='C')

ExTra = np.reshape(ExTra, (NFREQs, Nx, Ny), order='C')
EyTra = np.reshape(EyTra, (NFREQs, Nx, Ny), order='C')
EzTra = np.reshape(EzTra, (NFREQs, Nx, Ny), order='C')

ExInc = np.reshape(ExInc, (NFREQs, Nx, Ny), order='C')
EyInc = np.reshape(EyInc, (NFREQs, Nx, Ny), order='C')
EzInc = np.reshape(EzInc, (NFREQs, Nx, Ny), order='C')

kx = -2*math.pi*m/(Nx*dx)
ky = -2*math.pi*n/(Ny*dy)
print(len(EzInc))
for ff in range (0, NFREQs):

	Esr = 0
	lam = c0/freq[ff]
	# print(lam)
	k0 = 2*math.pi/lam
	kzinc = k0*nref

	for i in range (0, Nx):
		for j in range (0, Ny):
			kzR[i, j] = cmath.sqrt((k0*nref)**2 - kx[i]**2 - ky[j]**2)
			kzT[i, j] = cmath.sqrt((k0*ntra)**2 - kx[i]**2 - ky[j]**2)
	Esr = np.sqrt(abs(ExInc[ff])**2 + abs(EyInc[ff])**2 + abs(EzInc[ff])**2)#ExInc[ff]*conj(ExInc[ff]) + EyInc[ff]*conj(EyInc[ff]) + EzInc[ff]*conj(EzInc[ff]))
	
	Sxr = ExRef[ff]/Esr
	Syr = EyRef[ff]/Esr
	Szr = EzRef[ff]/Esr

	Sxt = ExTra[ff]/Esr
	Syt = EyTra[ff]/Esr
	Szt = EzTra[ff]/Esr

	# print(Sxr)
	
	Sxrfft = fftshift(fft2(Sxr))/(Nx*Ny)
	Syrfft = fftshift(fft2(Syr))/(Nx*Ny)
	Szrfft = fftshift(fft2(Szr))/(Nx*Ny)

	Sxtfft = fftshift(fft2(Sxt))/(Nx*Ny)
	Sytfft = fftshift(fft2(Syt))/(Nx*Ny)
	Sztfft = fftshift(fft2(Szt))/(Nx*Ny)

	Sref = abs(Sxrfft)**2 + abs(Syrfft)**2 + abs(Szrfft)**2 #conj(Sxrfft)*Sxrfft + conj(Syrfft)*Syrfft + conj(Szrfft)*Szrfft
	Stra = abs(Sxtfft)**2 + abs(Sytfft)**2 + abs(Sztfft)**2 #conj(Sxtfft)*Sxrfft + conj(Sytfft)*Syrfft + conj(Sztfft)*Szrfft

	ref[ff] = Sref*(real(kzR/kzinc))
	tra[ff] = Stra*(real(kzT/kzinc))

for ff in range(0, NFREQs):
	for j in range(0, Ny):
		for i in range(0, Nx):
			REFPaperAg[ff] =  REFPaperAg[ff] + ref[ff, i, j]
			TRAPaperAg[ff] =  TRAPaperAg[ff] + tra[ff, i, j]


###########################################################################
###########################  Palik Scan Ag ################################
###########################################################################
freq = np.zeros(NFREQs, dtype = np.double)
m  = np.zeros(Nx, dtype = np.int)
n  = np.zeros(Ny, dtype = np.int)

kx = np.zeros(Nx, dtype = np.double)
ky = np.zeros(Ny, dtype = np.double)

Sxr    = np.zeros((Nx, Ny), dtype = np.complex)
Syr    = np.zeros((Nx, Ny), dtype = np.complex)

kzR    = np.zeros((Nx, Ny), dtype = np.complex)
kzT    = np.zeros((Nx, Ny), dtype = np.complex)

Esr    = np.zeros((Nx, Ny), dtype = np.complex)

EyIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex) 
EyIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncR = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzIncI = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)


ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REFScan   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRAScan   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

freq   = np.loadtxt("ScannedAg/freq.txt",  usecols=(0), skiprows= 0, unpack =True)

ExR    = np.loadtxt("ScannedAg/ExRef.txt", usecols=(0), skiprows= 1, unpack =True)
ExI    = np.loadtxt("ScannedAg/ExRef.txt", usecols=(1), skiprows= 1, unpack =True)

EyR    = np.loadtxt("ScannedAg/EyRef.txt", usecols=(0), skiprows= 1, unpack =True)
EyI    = np.loadtxt("ScannedAg/EyRef.txt", usecols=(1), skiprows= 1, unpack =True)

EzR    = np.loadtxt("ScannedAg/EzRef.txt", usecols=(0), skiprows= 1, unpack =True)
EzI    = np.loadtxt("ScannedAg/EzRef.txt", usecols=(1), skiprows= 1, unpack =True)

ExTR   = np.loadtxt("ScannedAg/ExTra.txt", usecols=(0), skiprows= 1, unpack =True)
ExTI   = np.loadtxt("ScannedAg/ExTra.txt", usecols=(1), skiprows= 1, unpack =True)

EyTR   = np.loadtxt("ScannedAg/EyTra.txt", usecols=(0), skiprows= 1, unpack =True)
EyTI   = np.loadtxt("ScannedAg/EyTra.txt", usecols=(1), skiprows= 1, unpack =True)

EzTR   = np.loadtxt("ScannedAg/EzTra.txt", usecols=(0), skiprows= 1, unpack =True)
EzTI   = np.loadtxt("ScannedAg/EzTra.txt", usecols=(1), skiprows= 1, unpack =True)

ExIncR = np.loadtxt("ScannedAg/ExInc.txt", usecols=(0), skiprows= 0, unpack =True)
ExIncI = np.loadtxt("ScannedAg/ExInc.txt", usecols=(1), skiprows= 0, unpack =True)

EyIncR = np.loadtxt("ScannedAg/EyInc.txt", usecols=(0), skiprows= 0, unpack =True)
EyIncI = np.loadtxt("ScannedAg/EyInc.txt", usecols=(1), skiprows= 0, unpack =True)

EzIncR = np.loadtxt("ScannedAg/EzInc.txt", usecols=(0), skiprows= 0, unpack =True)
EzIncI = np.loadtxt("ScannedAg/EzInc.txt", usecols=(1), skiprows= 0 , unpack =True)

ExRef = ExR + 1j*ExI
EyRef = EyR + 1j*EyI
EzRef = EzR + 1j*EzI

ExTra = ExTR + 1j*ExTI
EyTra = EyTR + 1j*EyTI
EzTra = EzTR + 1j*EzTI

ExInc = (ExIncR + 1j*ExIncI)
EyInc = (EyIncR + 1j*EyIncI)
EzInc = (EzIncR + 1j*EzIncI)

ExRef = np.reshape(ExRef, (NFREQs, Nx, Ny), order='C')
EyRef = np.reshape(EyRef, (NFREQs, Nx, Ny), order='C')
EzRef = np.reshape(EzRef, (NFREQs, Nx, Ny), order='C')

ExTra = np.reshape(ExTra, (NFREQs, Nx, Ny), order='C')
EyTra = np.reshape(EyTra, (NFREQs, Nx, Ny), order='C')
EzTra = np.reshape(EzTra, (NFREQs, Nx, Ny), order='C')

ExInc = np.reshape(ExInc, (NFREQs, Nx, Ny), order='C')
EyInc = np.reshape(EyInc, (NFREQs, Nx, Ny), order='C')
EzInc = np.reshape(EzInc, (NFREQs, Nx, Ny), order='C')

kx = -2*math.pi*m/(Nx*dx)
ky = -2*math.pi*n/(Ny*dy)
print(len(EzInc))
for ff in range (0, NFREQs):

	Esr = 0
	lam = c0/freq[ff]
	# print(lam)
	k0 = 2*math.pi/lam
	kzinc = k0*nref

	for i in range (0, Nx):
		for j in range (0, Ny):
			kzR[i, j] = cmath.sqrt((k0*nref)**2 - kx[i]**2 - ky[j]**2)
			kzT[i, j] = cmath.sqrt((k0*ntra)**2 - kx[i]**2 - ky[j]**2)
	Esr = np.sqrt(abs(ExInc[ff])**2 + abs(EyInc[ff])**2 + abs(EzInc[ff])**2)#ExInc[ff]*conj(ExInc[ff]) + EyInc[ff]*conj(EyInc[ff]) + EzInc[ff]*conj(EzInc[ff]))
	
	Sxr = ExRef[ff]/Esr
	Syr = EyRef[ff]/Esr
	Szr = EzRef[ff]/Esr

	Sxt = ExTra[ff]/Esr
	Syt = EyTra[ff]/Esr
	Szt = EzTra[ff]/Esr

	# print(Sxr)
	
	Sxrfft = fftshift(fft2(Sxr))/(Nx*Ny)
	Syrfft = fftshift(fft2(Syr))/(Nx*Ny)
	Szrfft = fftshift(fft2(Szr))/(Nx*Ny)

	Sxtfft = fftshift(fft2(Sxt))/(Nx*Ny)
	Sytfft = fftshift(fft2(Syt))/(Nx*Ny)
	Sztfft = fftshift(fft2(Szt))/(Nx*Ny)

	Sref = abs(Sxrfft)**2 + abs(Syrfft)**2 + abs(Szrfft)**2 #conj(Sxrfft)*Sxrfft + conj(Syrfft)*Syrfft + conj(Szrfft)*Szrfft
	Stra = abs(Sxtfft)**2 + abs(Sytfft)**2 + abs(Sztfft)**2 #conj(Sxtfft)*Sxrfft + conj(Sytfft)*Syrfft + conj(Sztfft)*Szrfft

	ref[ff] = Sref*(real(kzR/kzinc))
	tra[ff] = Stra*(real(kzT/kzinc))

for ff in range(0, NFREQs):
	for j in range(0, Ny):
		for i in range(0, Nx):
			REFScan[ff] =  REFScan[ff] + ref[ff, i, j]
			TRAScan[ff] =  TRAScan[ff] + tra[ff, i, j]

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
plt.xlim(0,10)
# plt.ylim(0,1)

plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{}$', color='black', linewidth = 5)


# plt.plot((c0/freq)*1e6, REFOur, label = r'$\rm R_{XFDTD \ Our \ Model}$', color = "red", linewidth = 3)
# plt.plot((c0/freq)*1e6, REFPaperAg, label = r'$\rm R_{XFDTD \ Paper \ Model}$', color = "orange", linewidth = 3)
plt.plot((c0/freq)*1e6, REFScan, label = r'$\rm R_{XFDTD \ Scanned \ Palik \ Data}$', color = "red", linewidth = 3)

# plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{XFDTD}$', color = "black", linewidth = 4)
# plt.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{XFDTD}$', color = "limegreen", linewidth = 2)


plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='center right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
# plt.savefig("XFAgFilmRTTheoryDDForm.pdf")
plt.savefig("XFAginVac.png")

# plt.show()
