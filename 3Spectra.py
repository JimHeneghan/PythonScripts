from numpy import *
from scipy import *
from pylab import *
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
#in k
eps_inf = 4.87
#S_nu = 1.83
c = 3e8
omega_TO = 137000
omega_LO = 161000
gamma_nu = 500
gamma_nuJJ = (1.32e12)/(2*math.pi*c)

k = linspace(50000, 200000, 2000)
#print omega
epsk = eps_inf*( 1 + (omega_LO**2 - omega_TO**2)/(omega_TO**2-1j*gamma_nu*k - k**2))
epskJJ = eps_inf*( 1 + (omega_LO**2 - omega_TO**2)/(omega_TO**2-1j*gamma_nuJJ*k - k**2))

eps_infz = 2.95
#S_nuz = 0.61
omega_TOz = 78000
omega_LOz = 83000
gamma_nuz = 400

epskz = eps_infz*( 1 + (omega_LOz**2 - omega_TOz**2)/(omega_TOz**2-1j*gamma_nuz*k - k**2))

c=3e8
gamma_nuz2 = (3.97e11)/(2*math.pi*c)

epskz2 = eps_infz*( 1 + (omega_LOz**2 - omega_TOz**2)/(omega_TOz**2-1j*gamma_nuz2*k - k**2))
fig, ax = plt.subplots(3, figsize = (12, 12),constrained_layout=True)

######### Theory Cmparison plot #######
d = 84.0e-9

# Load Complex Dielectric Function Data
lambda1, n_r, n_i = loadtxt("hBNgetindexConstZ.txt", usecols=(0,1,2),  unpack = True)

# Calculate Complex Refractive Index
nc = n_r + 1j*n_i
eps = nc**2
# Calculate Complex Propagation Constant
k0 = 2.0*pi/(lambda1*1.0e-6)
k1=k0*nc

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below
T = abs(1.0/(cos(k1*d)-1j*(nc+1.0/nc)*sin(k1*d)/2.0))**2# Load FDTD Transmission Data

lambda2, TFDTD= loadtxt("hBN_RTConstZ32.txt", usecols=(0,2,),  unpack = True)

# Formatting and Plotting
#ax[0].rc('axes', linewidth=2)
ax[2].tick_params(width=2, labelsize=20)
ax[2].set_xlim(500, 2000)
#xlim(5,10)
ax[2].set_ylabel("T", fontsize = '30')
plt.setp(ax[2].spines.values(), linewidth=2)
#xlabel(r'$ \rm \lambda \ (\mu m)$', fontsize = '30')

ax[2].plot(10000/(lambda1), T, label=r'$\rm T_{Orfanidis}$', color='red', linewidth = 3)
ax[2].plot(10000/(lambda2), TFDTD,  label=r'$ \rm T_{Lumerical}$', color='blue' )
ax[2].legend(loc='upper left', fontsize='20')


######## Real Permittivity Comparrison Plot ############
ax[0].tick_params(width=2, labelsize=20)
ax[0].set_xlim(500, 2000)
plt.setp(ax[0].spines.values(), linewidth=2)

ax[0].plot(k/100, real(epsk),  color = "red", linewidth=3, label = r"$\epsilon_{\perp} $ Zhao")
ax[0].plot(k/100, real(epskJJ),  color = "black" ,label = r"$\epsilon_{\perp} $ Jiang")
ax[0].plot(k/100, real(epskz), color = "red",  linewidth=3, label = r"$\epsilon_{\parallel}$ Zhao")
ax[0].plot(k/100, real(epskz2), color = "black", label = r"$\epsilon_{\parallel}$ Jiang")
ax[0].set_ylabel(r"Re$(\epsilon)$", fontsize = '30')
#ax[0].set_xlabel(r"$\rm Frequency cm^{-1}$", fontsize = '30')
ax[0].legend(loc= 'upper right', fontsize='15')
# plt.savefig("/halhome/jimheneghan/UsefulImages/ReEpsComparissonbtnJiang&ZZ.pdf")
# plt.savefig("/halhome/jimheneghan/UsefulImages/ReEpsComparissonbtnJiang&ZZ.png")
#show()

######## Imaginary Permittivity Comparrison Plot ############

#fig, ax = plt.subplots(figsize = (10, 6),constrained_layout=True)
ax[1].tick_params(width=2, labelsize=20)
ax[1].set_xlim(500, 2000)
plt.setp(ax[1].spines.values(), linewidth=2)
ax[1].plot(k/100, imag(epsk), color = "red", linewidth=3, label = r"$\epsilon_{\perp}$ Zhao" )
ax[1].plot(k/100, imag(epskJJ), color = "black" , label = r"$\epsilon_{\perp} $ Jiang")
ax[1].plot(k/100, imag(epskz), color = "red", linewidth=3, label = r"$\epsilon_{\parallel}$ Zhao")
ax[1].plot(k/100, imag(epskz2), color = "black" , label = r"$\epsilon_{\parallel}$ Jiang")
ax[1].set_ylabel(r"Im$(\epsilon)$", fontsize = '30')
ax[2].set_xlabel(r"$\rm Frequency \ cm^{-1}$", fontsize = '30')
ax[1].legend(loc= 'upper left', fontsize='18')
plt.savefig("/halhome/jimheneghan/UsefulImages/ImEpsReEpsTSpectraComboFeb19.pdf")
plt.savefig("/halhome/jimheneghan/UsefulImages/ImEpsReEpsTSpectraComboFeb19.png")
show()
