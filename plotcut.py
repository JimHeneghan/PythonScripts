from pylab import *
#x, y = loadtxt('dispersion.txt', usecols=(0,1,), unpack=True, skiprows=0)
#plot(x,y)
#scatter(59843.647483298715,1481.4814814814813, s=20, c='r')
#scatter(52204.032910962706,1449.2753623188405, s=20, c='r')
#scatter(65723.640167364,1449.2753623188405, s=20, c='b')
#show()
lam = 6.9

# Set the threshold for the peak finder
threshold = 2.5

# Set the amount of padding
numpad = 100000

# Storage for peak data
peakk = []
peakfe = []

# Load data
x, e = loadtxt('6.9_um.txt', usecols=(0,1,), unpack=True, skiprows=3)

# Scale of the x dat was incorrect by a factor of a 100


# Zero pad so that the FFT is smooth
pe = pad(e, numpad, mode='constant')

# Calculate the range of spatial frequencies
n=pe.size
dx = x[1]
dk = 1/(n*dx)
k=arange(0,n*dk,dk)

# Fourier transform data and take absolute value
fe = abs(fft(pe))

print("Excitation Frequency", 1/(lam/1e4), "cm^-1")

# Find peaks and store peak data
for i in range(2,n//2):
    if ((fe[i] > fe[i-1]) & (fe[i] > fe[i+1]) & (fe[i] > threshold)):
        print(fe[i], 1.0/k[i], "um", 2*pi*k[i]*1e4, "cm-1")
        peakk.append(k[i])
        peakfe.append(fe[i])


# Plot Fourier transformed data and peaks
#plot(x,e)
#show()
plot(k[:n//2],fe[:n//2])
scatter(peakk, peakfe, s=20, c='r')		
show()


