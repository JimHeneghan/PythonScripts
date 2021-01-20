from scipy import *
from pylab import *
import matplotlib.pyplot as plt
# Thickness of hBN Film
d = 83.0e-9

# Load Complex Dielectric Function Data
lambda1, n_r, n_i = loadtxt("hBNgetindexConstZ.txt", usecols=(0,1,2),  unpack = True)

# Calculate Complex Refractive Index
nc = n_r + 1j*n_i
eps = nc**2


# Calculate Complex Propagation Constant
k0 = 2.0*pi/(lambda1*1.0e-6)
k=k0*nc

# Calculate (Power) Reflection from the result of problem equation 5.4.3 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below
#T = abs(1.0/(cos(k*d)-1j*(nc+1.0/nc)*sin(k*d)/2.0))**2# Load FDTD Transmission Data

eta0 = 1
eta1 = 1/nc
eta2 = 1

ro1 = (eta1 - eta0)/(eta1 + eta0)

ro2 = (eta2 - eta1)/(eta2 + eta1)

R = abs((ro1 + ro2*np.exp(2*1j*k*d))/(1 + ro1*ro2*np.exp(2*1j*k*d)))

lambda2, RFDTD= loadtxt("hBN_RTConstZ32.txt", usecols=(0,1,),  unpack = True)

# Formatting and Plotting
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)

xlim(5,10)
ylabel("T", fontsize = '30')
xlabel(r'$\lambda (\mu m)$', fontsize = '30')

plot(lambda1, R, label=r'$R_{Orfanidis}$', color='red')
plot(lambda2, RFDTD,  label=r'$R_{FDTD: Lumerical}$', color='blue')
plt.tight_layout()
legend(loc='upper right', fontsize='15')

plt.show()
