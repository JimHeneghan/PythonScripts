from scipy import *
from pylab import *

# Thickness of hBN Film
d = 83.0e-9

# Load Complex Dielectric Function Data
lambda1, ReEps, ImEps = loadtxt("hBNgetmaterial.txt", usecols=(0,1,2),  unpack = True)

# Calculate Complex Refractive Index
eps=ReEps + 1j*ImEps
nc=sqrt(eps)

# Calculate Complex Propagation Constant
k0 = 2.0*pi/(lambda1*1.0e-6)
k=k0*nc

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below
T=abs(1.0/(cos(k*d)-1j*(nc+1.0/nc)*sin(k*d)/2.0))**2

# Load FDTD Transmission Data
lambda2, TFDTD= loadtxt("hBN_RTAug22.txt", usecols=(0,2,),  unpack = True)

# Formatting and Plotting
rc('axes', linewidth=2)
tick_params(width=2, labelsize=20)

xlim(5,10)
ylabel("T", fontsize = '30')
xlabel(r'$\lambda (\mu m)$', fontsize = '30')

plot(lambda1, T, label=r'$T_{Theory}$', color='red')
plot(lambda2, TFDTD,  label=r'$T_{FDTD:getmaterial}$', color='blue' )
legend(loc='upper left', fontsize='20')

show()
