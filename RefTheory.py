from scipy import *
from pylab import *

# Thickness of hBN Film
d = 84.0e-9

# Load Complex Dielectric Function Data
lambda1, n_r, n_i = loadtxt("hBNgetindexConstZ.txt", usecols=(0,1,2),  unpack = True)

# Calculate Complex Refractive Index
nc = n_r + 1j*n_i
eps = nc**2


# Calculate Complex Propagation Constant
k0 = 2.0*pi/(lambda1*1.0e-6)
k=k0*nc

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below
T = abs(1.0/(cos(k*d)-1j*(nc+1.0/nc)*sin(k*d)/2.0))**2# Load FDTD Transmission Data

lambda2, TFDTD= loadtxt("hBN_RTConstZ32.txt", usecols=(0,2,),  unpack = True)

# Formatting and Plotting
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)

xlim(5,10)
ylabel("T", fontsize = '30')
xlabel(r'$ \rm \lambda \ (\mu m)$', fontsize = '30')

plot(lambda1, T, label=r'$\rm T_{Orfanidis}$', color='red', linewidth = 3)
plot(lambda2, TFDTD,  label=r'$ \rm T_{Lumerical}$', color='blue' )
legend(loc='lower right', fontsize='15')
plt.tight_layout()
plt.savefig("/halhome/jimheneghan/UsefulImages/OrfandisVLumericalTrans.png")
plt.savefig("/halhome/jimheneghan/UsefulImages/OrfandisVLumericalTrans.pdf")
plt.show()
