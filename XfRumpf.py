import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl

NFREQs = 500
c0 = 2.99792458e8

dx = 1e-8
dy = 1e-8
df = (9e-6)/NFREQs
Nx = 10
Ny = 10
Time = 6800
dt = 3.30278e-18
KRef     = np.zeros(NFREQs, dtype = np.complex)
ExRef   = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EyRef   = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzRef   = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)

ExInc    = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EyInc    = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)
EzInc    = np.zeros((NFREQs, Nx, Ny), dtype = np.complex)

freq = np.zeros(NFREQs, dtype = np.double)


# EyThing = np.zeros(Time, dtype = np.double)
Ex, Ey, Ez = np.loadtxt("Planardata2.txt",  usecols=(5,6,7), skiprows= 1, unpack =True)

# Ey    = np.loadtxt("Planardata2.txt",  usecols=(6), skiprows= 1, unpack =True)
Source = np.loadtxt("PlaneWaveData.txt",  usecols=(2), skiprows= 1, unpack =True) 	
Ex  = np.reshape(Ex, (Time, Nx, Ny), order='C')
Ey  = np.reshape(Ey, (Time, Nx, Ny), order='C')
Ez  = np.reshape(Ez, (Time, Nx, Ny), order='C')

ExRef  = np.reshape(ExRef, (NFREQs, Nx, Ny), order='C')
EyRef  = np.reshape(EyRef, (NFREQs, Nx, Ny), order='C')
EzRef  = np.reshape(EzRef, (NFREQs, Nx, Ny), order='C')

ExInc  = np.reshape(ExInc, (NFREQs, Nx, Ny), order='C')
EyInc  = np.reshape(EyInc, (NFREQs, Nx, Ny), order='C')
EzInc  = np.reshape(EzInc, (NFREQs, Nx, Ny), order='C')
# EInc   = np.reshape(Source, (Time, Nx, Ny), order='C')

# for t in range(0, Time):
# 	EyThing[t] = EyRef[t, 5, 5]

# plt.plot(EyThing)
# plt.show()
# print(EyRef)

for i in range (0, NFREQs):
	freq[i] = c0/(1e-6 + i*df)
	KRef[i] = np.exp(-1j*2*cmath.pi*dt*freq[i])


for t in range (0, Time):
	for i in range (0, Nx):
		for j in range (0, Ny):
			for f in range (0, NFREQs):
				# RefEx  = ExRef[t,i,j]
				# RefEy  = EyRef[t,i,j]
				# RefEz  = EzRef[t,i,j]

				ExRef[f, i, j] = ExRef[f, i, j] + (KRef[f]**t)*Ex[t,i,j]
				EyRef[f, i, j] = EyRef[f, i, j] + (KRef[f]**t)*Ey[t,i,j]
				EzRef[f, i, j] = EzRef[f, i, j] + (KRef[f]**t)*Ez[t,i,j]

				ExInc[f, i, j] = ExInc[f, i, j] + (KRef[f]**t)*Source[t]
				EyInc[f, i, j] = EyInc[f, i, j] + (KRef[f]**t)*Source[t]
				EzInc[f, i, j] = EzInc[f, i, j] + (KRef[f]**t)*Source[t]

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
			REF[ff] =  REF[ff] + ref[ff, i, j]
			TRA[ff] =  TRA[ff] + tra[ff, i, j]

fig, ax = plt.subplots(figsize=(15,9),constrained_layout=True)
plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(direction = 'in', width=2, labelsize=20)
plt.ylabel("R", fontsize = '30')   
plt.xlabel(r"$\rm Wavelength \ (\mu m)$", fontsize = '30')
# plt.xlim(0,10)
plt.ylim(0,1)

plt.plot((c0/freq)*1e6, REF, label = r'$\rm R_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "red", linewidth = 6)
plt.plot((c0/freq)*1e6, TRA, label = r'$\rm T_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "black", linewidth = 4)
# plt.plot((c0/freq)*1e6, (1-(REF+TRA)), label = r'$\rm A_{FDTD \ Ag Pattern \ Toy \ nm^{3}}$', color = "limegreen", linewidth = 2)

# plt.plot((c0*2*math.pi/(w))*1e6, Rm, label=r'$\rm R_{TM}$', color='yellow', linewidth = 2))

plt.setp(ax.spines.values(), linewidth=2)
plt.tick_params(left = False, bottom = False)   
ax.legend(loc='upper right', fontsize='22')
# ax.axhline(y =1, color = 'black')
# ax.axvline(x =0.1, color = 'black')
plt.savefig("XF_AgSlab.pdf")
plt.savefig("XF_AgSlab.png")

