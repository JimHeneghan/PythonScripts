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

# k0 = k*1e12
c0 = 3e8
dx = 1e-8
dy = 1e-8



Nx = 10
Ny = 10

NFREQs = 500
nref = 1.0
ntra = 1.0

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



ref   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
REF   = np.zeros(NFREQs, dtype = np.double)

tra   = np.zeros((NFREQs, Nx, Ny), dtype = np.double)
TRA   = np.zeros(NFREQs, dtype = np.double)
for i in range (0, Nx):
	m[i] = i - (Nx  - 1)/2	
print(m)
for i in range (0, Ny):
	n[i] = i - (Ny  - 1)/2

ExInc    = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EyInc    = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzInc    = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
# EyThing = np.zeros(Time, dtype = np.double)
ExR, ExI, EyR, EyI, EzR, EzI, = np.loadtxt("PlaneRefLong.txt",  usecols=(5,6,7,8,9,10), skiprows= 1, unpack =True)
freq, EyIncR, EyIncI = np.loadtxt("PlaneIncLong.txt",  usecols=(0,3,4), skiprows= 1, unpack =True)
freq = freq*1e12
for i in range (0, Nx):
	for j in range (0, Ny):
		for f in range (0, NFREQs):
			ExInc[f,i,j] = (EyIncR[f] + 1j*EyIncI[f])
			EyInc[f,i,j] = (EyIncR[f] + 1j*EyIncI[f])
			EzInc[f,i,j] = (EyIncR[f] + 1j*EyIncI[f])

ExRef = (ExR + 1j*ExI)
EyRef = (EyR + 1j*EyI)
EzRef = (EzR + 1j*EzI)


# print(EyInc)
# ExTra = ExTR + 1j*ExTI
# EyTra = EyTR + 1j*EyTI
# EzTra = EzTR + 1j*EzTI

# ExInc = (ExIncR + 1j*ExIncI)
# EyInc = (EyIncR + 1j*EyIncI)
# EzInc = (EzIncR + 1j*EzIncI)

ExRef = np.reshape(ExRef, (NFREQs, Nx, Ny), order='C')
EyRef = np.reshape(EyRef, (NFREQs, Nx, Ny), order='C')
EzRef = np.reshape(EzRef, (NFREQs, Nx, Ny), order='C')

# ExTra = np.reshape(ExTra, (NFREQs, Nx, Ny), order='C')
# EyTra = np.reshape(EyTra, (NFREQs, Nx, Ny), order='C')
# EzTra = np.reshape(EzTra, (NFREQs, Nx, Ny), order='C')

ExInc = np.reshape(ExInc, (NFREQs, Nx, Ny), order='C')
EyInc = np.reshape(EyInc, (NFREQs, Nx, Ny), order='C')
EzInc = np.reshape(EzInc, (NFREQs, Nx, Ny), order='C')
EyIThing = np.zeros(NFREQs, dtype = np.complex)
EyRThing = np.zeros(NFREQs, dtype = np.complex)
for i in range (0, Nx):
	for j in range (0, Ny):
		for f in range (0, NFREQs):
			EyIThing[f] = EyInc[f,5,5]
			EyRThing[f] = EyRef[f,5,5]


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
			# kzT[i, j] = cmath.sqrt((k0*ntra)**2 - kx[i]**2 - ky[j]**2)
	Esr = np.sqrt(abs(ExInc[ff])**2 + abs(EyInc[ff])**2 + abs(EzInc[ff])**2)#ExInc[ff]*conj(ExInc[ff]) + EyInc[ff]*conj(EyInc[ff]) + EzInc[ff]*conj(EzInc[ff]))
	
	Sxr = ExRef[ff]/Esr
	Syr = EyRef[ff]/Esr
	Szr = EzRef[ff]/Esr

	# Sxt = ExTra[ff]/Esr
	# Syt = EyTra[ff]/Esr
	# Szt = EzTra[ff]/Esr

	# print(Sxr)
	
	Sxrfft = fftshift(fft2(Sxr))/(Nx*Ny)
	Syrfft = fftshift(fft2(Syr))/(Nx*Ny)
	Szrfft = fftshift(fft2(Szr))/(Nx*Ny)

	# Sxtfft = fftshift(fft2(Sxt))/(Nx*Ny)
	# Sytfft = fftshift(fft2(Syt))/(Nx*Ny)
	# Sztfft = fftshift(fft2(Szt))/(Nx*Ny)

	Sref = abs(Sxrfft)**2 + abs(Syrfft)**2 + abs(Szrfft)**2 #conj(Sxrfft)*Sxrfft + conj(Syrfft)*Syrfft + conj(Szrfft)*Szrfft
	# Stra = abs(Sxtfft)**2 + abs(Sytfft)**2 + abs(Sztfft)**2 #conj(Sxtfft)*Sxrfft + conj(Sytfft)*Syrfft + conj(Sztfft)*Szrfft

	ref[ff] = Sref*(real(kzR/kzinc))
	# tra[ff] = Stra*(real(kzT/kzinc))

for ff in range(0, NFREQs):
	for j in range(0, Ny):
		for i in range(0, Nx):
			REF[ff] =  REF[ff] + ref[ff, i, j]
			# TRA[ff] =  TRA[ff] + tra[ff, i, j]

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0,10)
plt.ylim(0,1)

# plt.plot((c0/freq)*1e6, REF, label = r'$\rm R_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "red", linewidth = 6))
# plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "black", linewidth = 4)
# plt.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "limegreen", linewidth = 2)

# plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='yellow', linewidth = 2))


# plt.plot((c0/freq)*1e6, abs(EyIThing)**2, label = r'$\rm XF \ Inc$', color = "black", linewidth = 6)
plt.plot((c0/freq)*1e6, (abs(EyRThing)**2)/(abs(EyIThing)**2), label = r'$\rm XF \ Ref$', color = "red", linewidth = 6)

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
plt.savefig("LongRunSimpleSpec.pdf")
plt.savefig("LongRunSimpleSpec.png")

# plt.show()
