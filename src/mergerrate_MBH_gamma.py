import numpy as np
import numpy.random as nr
import scipy.optimize as so
import scipy.integrate as si
import matplotlib.pyplot as plt

# Spectrum Computation Functions
Gu = 6.673848e-11  # gravitational constant in m^3 kg^-1 s^-2
cu = 2.997925e8  # speed of light in m/s
Msun = 1.98855e30  # mass of the sun in kg
pc = 3.0856776e16  # parsec in meters
Gyr = 3.15576e16  # Gyr in seconds
G = Gu * Msun / pc**3.0  # gravitational parameter
c = cu / pc  # speed of light in parsecs/s

def sig(m):
    """Compute the mass-dependent sigma value."""
    return 261.5 * (m / 1.0e9)**(1.0 / 4.38) * 1000.0 / pc

def ms(m):
    """Compute the mass scaling function."""
    return 3.22e3 * m**0.862

def at(ms, gamma):
    """Calculate the scaling factor based on stellar mass and gamma."""
    return 2.95 * (ms / 1.0e6)**0.596 * (2.0**(1.0 / (3.0 - gamma)) - 1.0) / 0.7549

def ft(H, rho, mc, e):
    """Compute the f(t) function for given parameters."""
    return 0.56 / (np.pi * G**0.1) * (5.0 / 64.0 * c**5.0 * H * rho / sig(2.5 * mc) / fe(e))**0.3 * mc**(-0.4)

def rhoi(gamma, m):
    """Calculate the density of the galaxy based on mass and gamma."""
    return (3.0 - gamma) * ms(m) / (4.0 * np.pi * at(ms(m), gamma)**3.0) * (2.0 * m / ms(m))**(gamma / (gamma - 3.0))

def xm(e0):
    """Calculate the x(m) function based on eccentricity."""
    return 1293.0 / 181.0 * (e0**(12.0 / 19.0) / (1.0 - e0**2.0) * (1.0 + 121.0 * e0**2.0 / 304.0)**(870.0 / 2299.0))**1.5

def g(f):
    """Compute the g(f) function for frequency f."""
	return 1.0e-15*(2.913*(f*1.0e8)**0.254*np.exp(-0.807*f*1.0e8)+74.24*(f*1.0e8)**1.77*np.exp(-3.7*f*1.0e8)+4.8*(f*1.0e8)**(-0.676)*np.exp(-0.6/(f*1.0e8)))

def inpl(e, f, f0):
    """Interpolate the input function based on eccentricity and frequency."""
    return g(f * (1.0e-10 / f0) * (xm(0.9) / xm(e))) * (1.0e-10 * xm(0.9) / f0 / xm(e))**(2.0 / 3.0)

def nm(mc, e0, f, ga, H):
    """Calculate the number of mergers based on the chirp mass and frequency."""
    return (inpl(e0, f, ft(H, 10.0**ga * rhoi(1.0, 2.5 * 10.0**mc), 10.0**mc, e0)))**2.0 / 585080 * (500.0 / 0.7)**3.0 * (10.0**mc / 4.166e8)**(5.0 / 3.0)

def nz(z):
    """Compute the redshift function."""
    return (1.02 / (1.0 + z))**(1.0 / 3.0)

def fe(e):
    """Calculate the f(e) function based on eccentricity."""
    return (1.0 + (73.0 / 24.0) * e**2.0 + (37.0 / 96.0) * e**4.0) / (1.0 - e**2.0)**3.5



# Cosmological Functions
def invE(z):
    """Compute the inverse of the Hubble parameter for given redshift z in a Lambda CDM cosmology."""
    OmegaM = 0.3
    Omegak = 0.0
    OmegaLambda = 0.7
    return 1.0 / np.sqrt(OmegaM * (1.0 + z)**3 + Omegak * (1.0 + z)**2 + OmegaLambda)

def dtdz(z):
    """Calculate the time differential with respect to redshift z."""
    t0 = 14.0  # age of the universe in Gyr
    if z == -1.0:
        z = -0.99  # avoid division by zero
    return t0 / (1.0 + z) * invE(z)

def dVcdz(z):
    """Calculate the comoving volume element per unit redshift."""
    c = 3.e8  # speed of light in m/s
    H0 = 70.e3  # Hubble constant in m/s/Mpc
    return 4.0 * np.pi * c / H0 * invE(z) * DM(z)**2

def DM(z):
    """Compute the comoving distance to redshift z."""
    c = 3.e8
    H0 = 70.e3
    return c / H0 * si.quad(invE, 0., z)[0]

def mchirpq(q):
    """Calculate the chirp mass function for mass ratio q."""
    return np.log10(q**0.6 / (1.0 + q)**0.2)

def mchirp(m1, m2):
    """Compute the chirp mass from two individual masses m1 and m2."""
    return (m1 * m2)**0.6 / (m1 + m2)**0.2

def mfraction(m):
    """Return the mass fraction based on the mass m with stellar-bulge mass relation."""
    if m > 10.0:
        return np.sqrt(6.9) * np.exp(-6.9 / (2.0 * (m - 10.0))) / (m - 10.0)**1.5 + 0.615
    else:
        return 0.615

def mfractionp(m):
    """Calculate the derivative of the mass fraction."""
    if m > 10.0:
        return np.exp(-6.9 / (2.0 * (m - 10.0))) / (m - 10.0)**3.5 / 10.0**m * (3.94 - 1.7 * (m - 10.0))
    else:
        return 0.0



# Class to compute merger rates and charactheristic spectrum 

class mergerrate(object):

	def __init__(self, M1, q, zp, f, Phi0, PhiI, M0, alpha0, alphaI, f0, alphaf, betaf, gammaf,
                     t0, alphatau, betatau, gammatau, alphastar, betastar, gammastar, epsilon, e0, rho0, zval):
		"""
        Initialize the merger rate parameters.
        
        Parameters:
        - M1: Mass of primary galaxy
        - q: Mass ratio between galaxies
        - zp: Redshift
        - f: Frequency of the spectrum
        - Phi0, PhiI: Galaxy stellar mass function renormalization rate
        - M0: Scale mass
        - alpha0, alphaI: Galaxy stellar mass function slope
        - f0: Pair fraction parameters
        - t0: Merger time scale parameters
        - alphastar, betastar, gammastar, epsilon: Mbulge-MBH relation parameters
        - e0: Eccentricity of the binaries
        - rho0: Galaxy density parameter
        - zval: Maximum redshift
        """
        # Initialize parameters
		self.M1 = 10.**M1
		self.M1diff = (M1.max()-M1.min())/(len(M1)-1.)/2.
		self._M1red = np.zeros(len(M1))
		self.MBH1 = np.zeros((len(M1),len(zp)))

        # Assign parameters to class attributes
		self._Phi0 = Phi0
		self._PhiI = PhiI
		self._M0 = 10.**M0
		self._alpha0 = alpha0
		self._alphaI = alphaI
		self.alphastar = alphastar
		self.betastar = betastar
		self.gammastar = gammastar
		self.zp = zp
		self.q = q
		self.qdiff = (q.max()-q.min())/(len(q)-1.)/2.
		self.f = 10.**f
		self.f0 = f0
		self.alphaf = alphaf
		self.betaf = betaf
		self.gammaf = gammaf
		self.t0 = t0
		self.alphatau = alphatau
		self.betatau = betatau
		self.gammatau = gammatau
		self.sigma = epsilon
		self.twosigma2 = 2.*epsilon*epsilon
		self.e0 = e0
		self.rho0 = rho0

		
        # Initialize derived parameters
		self.zpdiff = (zp.max()-zp.min())/(len(zp)-1.)/2.
		self._f0 = f0/self._fpairtest()
		self._beta1 = betaf - betatau
		self._gamma1 = gammaf - gammatau

		for i in range(len(M1)):
			self._M1red[i] = 10.**M1[i]*mfraction(M1[i]) 
			for j2 in range(len(zp)):
				self.MBH1[i,j2] = self.MBH(self._M1red[i],zp[j2]) 

		self.MBH1diff = (self.MBH1.max(0)-self.MBH1.min(0))/(len(self.MBH1)-1.)/2.
		self.M2 = np.zeros((len(M1),len(q)))
		self._M2red = np.zeros((len(M1),len(q)))
		self.MBH2 = np.zeros((len(M1),len(q),len(zp)))
		self.qBH = np.zeros((len(M1),len(q),len(zp)))
		self.Mc = np.zeros((len(M1),len(q),len(zp)))

        # Compute mass ratios and black hole masses
		for i,j in np.ndindex(len(M1),len(q)):
			self.M2[i,j] = self.M1[i]*q[j]
			self._M2red[i,j] = self.M1[i]*q[j]*mfraction(np.log10(self.M1[i]*q[j]))
			for j2 in range(len(zp)):
				self.MBH2[i,j,j2] = self.MBH(self._M2red[i,j],zp[j2]) 
				self.qBH[i,j,j2] = 10.**self.MBH2[i,j,j2]/10.**self.MBH1[i,j2]
				self.Mc[i,j,j2] = self.MBH1[i,j2]+mchirpq(self.qBH[i,j,j2])
		
	
	def zprime(self,M1,q,zp):
		"""
        Calculate the redshift condition. 
        This method improves the cut at the age of the universe.
		"""
        # Calculate time to redshift and merger time scale
		t0 = si.quad(dtdz,0,zp)[0]
		tau0 = self.tau(M1,q,zp)
		if t0+tau0 < 13.:
            # Solve for the adjusted redshift
			result = so.fsolve(lambda z: si.quad(dtdz,0,z)[0] - self.tau(M1,q,z) - t0,0.)[0]
		else:
			result = -1. # Condition not met

		return result

	def Phi(self,M1,Phi0,alpha0):
		"""
        Calculate the galaxy stellar mass function.
		"""
		return np.log(10.)*Phi0*(M1/self._M0)**(1.+alpha0)*np.exp(-M1/self._M0)

	def curlyF(self,M1,q,z):
		"""
        Calculate the galaxy pair fraction.
		"""
		return self._f0*(1.+z)**self.betaf*(M1/1.e11)**(self.alphaf)*q**(self.gammaf)

	def _fpairtest(self,qlow=0.25,qhigh=1.):
		"""
        Normalize the galaxy pair fraction.
		"""
		return (qhigh**(self.gammaf+1.)-qlow**(self.gammaf+1.))/(self.gammaf+1.)

	def tau(self,M1,q,z): 
		"""
        Calculate the merger time scale.
		"""
		return self.t0*(M1/5.7e10)**(self.alphatau)*(1.+z)**self.betatau*q**self.gammatau

	def dnGdM(self,M,n2,alpha1):
		"""
        Calculate the differential number density with respect to mass.
		"""
		return n2*self._M0*(M/self._M0)**alpha1*np.exp(-M/self._M0)

	def dnGdq(self,q):
		"""
        Calculate the differential number density with respect to mass ratio.
		"""
		return q**self._gamma1

	def dnGdz(self,z):
		"""
        Calculate the differential number density with respect to redshift.
		"""
		return (1.+z)**self._beta1*dtdz(z)

	def dndM1par(self,M1,q,z,n2,alpha1):
		"""
        Calculate the differential number density with respect to M1, q, and z.
		d3n/dM1dqdz from parameters
		"""
		return n2*self._M0*(M1/self._M0)**alpha1*np.exp(-M1/self._M0)*q**self._gamma1*(1.+z)**self._beta1*dtdz(z)*(0.4/0.7*1.e11/self._M0)**self.alphatau/(1.e11/self._M0)**self.alphaf

	def dndMc(self,M1,q,z,n2,alpha1):
		"""
        Calculate the number density of mergers per unit mass ratio and redshift.
		d3n/dMcdqdz
		"""
		Mred = M1*mfraction(np.log10(M1))

		return self.dndM1par(M1,q,z,n2,alpha1)/self.dMbdMG(M1)/self.dMBHdMG(Mred,z)*10.**self.MBH(Mred,z)*np.log(10.)

	def dMBHdMG(self,M,z): 
		"""
        Calculate the derivative of black hole mass with respect to galaxy mass.
		dMBH/dMG
		"""
		return self.alphastar/1.e11*(M/1.e11)**(self.alphastar-1.)*10**(self.betastar+self.gammastar*z)

	def dMbdMG(self,M):
		"""
        Calculate the derivative of bulge mass with respect to galaxy mass.
		dMbulge/dMG
		"""
		return mfraction(np.log10(M))+M*mfractionp(np.log10(M))

	def dndz(self,M1,q,z,n2,alpha1):
		"""
        Calculate the number density of mergers per redshift bin.
		dn/dz
		"""
		Mlow = 10.**(np.log10(M1)-self.M1diff)
		Mhigh = 10.**(np.log10(M1)+self.M1diff)
		qlow = q-self.qdiff
		qhigh = q+self.qdiff

		return n2*self._M0*(1.+z)**self._beta1*dtdz(z)*(qhigh**(self._gamma1+1.)-qlow**(self._gamma1+1.))/(self._gamma1+1.)*si.quad(lambda M: (M/self._M0)**alpha1*np.exp(-M/self._M0),Mlow,Mhigh)[0]*(0.4/0.7*1.e11/self._M0)**self.alphatau/(1.e11/self._M0)**self.alphaf

	def dNdtdz(self,M1,q,z,n2,alpha1,zp):
		"""
        Calculate the number of mergers per unit time per unit logarithmic redshift.
		dN/dtdlog_10 z
		"""
		return self.dndz(M1,q,z,n2,alpha1)/dtdz(zp)*dVcdz(zp)/1.e9*zp*np.log(10.)

	def MBH(self,M,z):
		"""
        Calculate the mass of the black hole based on the mass of the bulge and redshift.
		"""
		return  self.alphastar*np.log10(M/1.e11)+self.betastar+self.gammastar*z

	def dndzint(self,M1,q,z,n2,alpha1):
		"""
        Calculate the number of mergers per Gyr integrated over M1, q, and z.
		"""
		scale = (0.4/0.7*1.e11/self._M0)**self.alphatau/(1.e11/self._M0)**self.alphaf
		numberM = si.quad(self.dnGdM,10.**(np.log10(M1)-self.M1diff),10.**(np.log10(M1)+self.M1diff),args=(n2,alpha1))[0]
		numberq = si.quad(self.dnGdq,q-self.qdiff,q+self.qdiff)[0]
		numberz = si.quad(self.dnGdz,z-self.zpdiff,z+self.zpdiff)[0]

		return numberM*numberq*numberz*scale







	def output(self,function='zprime'):
		"""
	    Calculate a 3D array of values based on specified function and input parameters.
		input 3 x 1d array M1,q,z
		output 3d array (M1,q,z) (galaxy mass, galaxy mass ratio, redshift) of values for function
		"""
		# Initialize output array with zeros
		output = np.zeros((len(self.M1),len(self.q),len(self.zp)))

		# Iterate through all combinations of indices for M1, q, and zp
		for i,j,k in np.ndindex(len(self.M1),len(self.q),len(self.zp)):
			# Compute the redshift using the zprime method
			z = self.zprime(self.M1[i],self.q[j],self.zp[k])

			# Handle non-physical cases where redshift is non-positive
			if z <= 0.:
				output[i,j,k] = 1.e-20
			else:
				# Calculate necessary parameters based on the computed redshift
				Phi0 = 10.**(self._Phi0+self._PhiI*z)
				alpha0 = self._alpha0+self._alphaI*z
				alpha1 = alpha0 + self.alphaf - self.alphatau
				n2 = Phi0*self._f0/self._M0/self._M0/self.t0

				# Calculate the output based on the specified function
				if function=='zprime':
					output[i,j,k] = z
				elif function=='fpair':
					output[i,j,k] = self.curlyF(self.M1[i],self.q[j],z)/self.q[j]**self.gammaf*self._fpairtest()
				elif function=='curlyF':
					output[i,j,k] = self.curlyF(self.M1[i],self.q[j],z)
				elif function=='tau':
					output[i,j,k] = self.tau(self.M1[i],self.q[j],z)
				elif function=='Phi':
					output[i,j,k] = self.Phi(self.M1[i],Phi0,alpha0)
				elif function=='dndM1':
					output[i,j,k] = self.Phi(self.M1[i],Phi0,alpha0)*self.curlyF(self.M1[i],self.q[j],z)/self.tau(self.M1[i],self.q[j],z)*dtdz(z)*4.*self.M1diff*self.qdiff
				elif function=='dndM1par':
					output[i,j,k] = self.dndM1par(self.M1[i],self.q[j],z,n2,alpha1)*4.*self.M1diff*self.qdiff*np.log(10.)*self.M1[i]
				elif function=='dndMc':
					output[i,j,k] = self.dndMc(self.M1[i],self.q[j],z,n2,alpha1)*4.*self.MBH1diff[k]*self.qdiff
				elif function=='dndz':
					output[i,j,k] = self.dndz(self.M1[i],self.q[j],z,n2,alpha1)
				elif function=='dNdtdz':
					output[i,j,k] = self.dNdtdz(self.M1[i],self.q[j],z,n2,alpha1,self.zp[k])
				elif function=='dndzint':
					output[i,j,k] = self.number(self.M1[i],self.q[j],z,n2,alpha1)
				else:
					raise UserWarning("output function not defined")
				
		return output

	def grid(self,n0=None,M1=None,M2=None,function='dndMc'):
		"""
		Construct a 3D grid array based on the input parameters and function.
		input 3d array n0, 2d array MBH1, 3d array MBH2
		output 3d array (Mcbh,qbh,z) (black hole chirp mass, black hole mass ratio, redshift) of values for function
		"""

		# Compute n0 using output if not provided
		if n0 is None:
			n0 = self.output(function)

		# Default M1 and M2 if not provided
		if M1 is None:
			M1 = 10.**self.MBH1
		if M2 is None:
			M2 = 10.**self.MBH2

		# Define chirp mass and mass ratio ranges
		Mcbh = np.linspace(5,11,30)
		qbh = np.linspace(0,1,10)

		# Compute differences for grid calculations
		Mcbhdiff = (Mcbh.max()-Mcbh.min())/(len(Mcbh)-1.)/2.
		qbhdiff = (qbh.max()-qbh.min())/(len(qbh)-1.)/2.

		# Initialize output array
		output = np.zeros((len(Mcbh),len(qbh),len(self.zp)))

		# Precompute Mc and q arrays
		Mc = np.zeros((len(M1),len(M2[0,:])))
		q = np.zeros((len(M1),len(M2[0,:])))

		# Calculate chirp mass (Mc) and mass ratio (q) for each combination
		for i,j in np.ndindex(len(M1),len(M2[0,:])):
			Mc[i,j] = np.log10(mchirp(M1[i],M2[i,j]))
			if M2[i,j] > M1[i]:
				q[i,j] = M1[i]/M2[i,j]
			else:
				q[i,j] = M2[i,j]/M1[i]
		
		# Fill in the output grid based on Mc and q values
		for i,j in np.ndindex(len(M1),len(M2[0,:])):
			for i0,j0 in np.ndindex(len(Mcbh),len(qbh)):
				if abs(Mc[i,j]-Mcbh[i0]) < Mcbhdiff and abs(q[i,j]-qbh[j0]) < qbhdiff:
					for k in range(len(self.zp)):
						output[i0,j0,k] += n0[i,j,k]/1.3
				else:
					pass

		return output

	
	def dispersion(self,function='dndMc'):
		"""
		Generate a dispersed 3D array based on input and specified function.
		#input 3d array n0, 2d array MBH1, 3d array MBH2
		#output 3d array (MBH1,MBH2,z) (black hole mass 1, black hole mass 2, redshift) of dispersed values for function
		"""
		n0 = self.output(function)

		# Define mass ratios for black holes
		M1 = np.linspace(5,11,30)
		qbh = np.linspace(1.e-10,1,10)
		
		# Precompute M2 based on M1 and qbh
		M2 = np.zeros((len(M1),len(qbh)))
		for i0,j0 in np.ndindex(len(M1),len(qbh)):
			M2[i0,j0] = np.log10(10.**M1[i0]*qbh[j0])

		# Initialize output array for dispersion
		output = np.zeros((len(M1),len(qbh),len(self.zp)))
		A = (M1[1]-M1[0])*(qbh[1]-qbh[0])
		
		# Calculate volume for dispersion
		for i,j in np.ndindex(len(self.MBH1),len(self.MBH2[0,:])):
			vol = np.zeros((len(M1),len(qbh),len(self.zp)))
			for i0,j0,k in np.ndindex(len(M1),len(qbh),len(self.zp)):
				n1 = np.exp(-(self.MBH1[i,k]-M1[i0])*(self.MBH1[i,k]-M1[i0])/self.twosigma2)
				n2 = np.exp(-(self.MBH2[i,j,k]-M2[i0,j0])*(self.MBH2[i,j,k]-M2[i0,j0])/self.twosigma2)
				vol[i0,j0,k] = n1*n2*A/(np.pi*self.twosigma2)
			
			# Normalize and calculate output values
			for i0,j0,k in np.ndindex(len(M1),len(qbh),len(self.zp)):
				output[i0,j0,k] += vol[i0,j0,k]/np.sum(vol)*n0[i,j,k]*len(self.zp)
		
		return self.grid(n0=output,M1=10.**M1,M2=10.**M2)
	
	def realization(self,function='dndMc'):
		"""
		Generate a realization of the black hole masses based on specified function.
		"""
		n0 = self.output(function)
		M1 = np.zeros(len(self.MBH1))
		for i in range (len(M1)):
			M1[i] = nr.normal(loc=self.MBH1[i],scale=self.sigma)

		M2 = np.zeros(self.MBH2.shape)
		for i,j in np.ndindex(len(self.MBH1),len(self.MBH2[0,:])):
			M2[i,j] = nr.normal(loc=self.MBH2[i,j],scale=self.sigma)

		return self.grid(n0=n0,M1=10.**M1,M2=10.**M2)
	
	def hdrop(self,n0,f,fbin):
		"""
		Apply the hdrop function to the input array based on frequency.
		"""
		if fbin is None:
			return n0
		
		c = 3.e8 # Speed of light in m/s
		G = 1.33e20 # Gravitational constant in appropriate units
		fhigh = f + fbin
		flow = f - fbin

		# Initialize an array for the processed values
		Mcbh = np.linspace(5,11,30)
		nin = np.zeros((n0.shape))

		for i,j,k in np.ndindex(n0.shape):
			nin[i,j,k] = n0[i,j,k]*dVcdz(self.zp[k])/dtdz(self.zp[k])*5./96./(2.*np.pi)**(8./3.)*c**5./(G*10.**Mcbh[i])**(5./3.)*3./8.*(flow**(-8./3.)-fhigh**(-8./3.))/Gyr*2.*self.zpdiff/(0.5+self.zp[k]/2.)**(11./3.)
		
		nout = n0
		nm = np.sum(nin,axis=(1,2))
		
		# Determine cutoff for dispersion
		number = 0
		for cut in range(len(nm)):
			if number < 1:
				number += nm[-cut-1]
			else:
				break
		
		# Set values below the cutoff to zero
		for i,j,k in np.ndindex(n0.shape):
			if i in range(cut-1):
				nout[-i-1,j,k] = 0.

		return nout

	def hmodelt(self,fbin=None):
		"""
		Compute the charctereistic strain values based on the input frequency bin.
		"""
		ga = self.rho0
		H = 15.0
		Mcbh = np.linspace(5,11,30)
		result = np.zeros(len(self.f))

		# Obtain the dispersion rate for the specified function
		rate = self.dispersion(function='dndMc')
		out = np.copy(rate)

		# Calculate the model over the frequency range
		for k in range(len(self.f)):
			n0 = self.hdrop(rate,self.f[k],fbin)
			n0 = np.sum(n0, axis=1)
			n = np.zeros((len(n0[:,0]),len(self.zp)))

			for i,j in np.ndindex(len(n0[:,0]),len(self.zp)):
				n[i,j] = n0[i,j]*nm(Mcbh[i],self.e0,self.f[k],ga,H)*nz(self.zp[j])*2.*self.zpdiff
			
			n = np.sum(n)
			result[k] = 0.5*np.log10(n)

		return result, out


		

		

# Main execution point
if __name__ == "__main__":

    """
    Parameters for galaxy stellar mass function and merger rates:
    
    - Phi0, PhiI: Normalization rates for the stellar mass function.
    - M0: Scale mass.
    - alpha0, alphaI: Slopes for the stellar mass function.
    
    - Pair fraction parameters:
      - f0: Rate of pairs.
      - alphaf: Mass power law for pair fraction.
      - betaf: Redshift power law for pair fraction.
      - gammaf: Mass ratio power law for pair fraction.
      
    - Merger time scale parameters:
      - t0: Time scale for mergers.
      - alphatau: Mass power law for merger time scale.
      - betatau: Redshift power law for merger time scale.
      - gammatau: Mass ratio power law for merger time scale.
      
    - Black hole relation parameters:
      - alphastar, betastar, gammastar: Coefficients for the relation between stellar mass and black hole mass.
      - epsilon: Additional parameter for the relation.
      
    - e0, rho0, z: Additional parameters and redshift value.
	"""


    # Initialize mass ranges and parameters
    M1 = np.linspace(9, 12, 30)  # Stellar mass range
    q = np.linspace(0.25, 1, 15)  # Mass ratios
    mc = np.linspace(5, 11, 30)    # Chirp mass range
    mcdiff = (mc[1] - mc[0]) / 2.  # Half the difference for chirp mass
    f = np.linspace(-9, -6, 15)    # Frequency range for analysis
	# f = np.linspace(-10,-7,15)

    # Define parameter names for plotting and reference
    namesr = [r'$\Phi_0$', r'$\Phi_I$', r'$M_0$', r'$\alpha_0$', r'$\alpha_I$', 
              r'$f_0$', r'$\alpha_f$', r'$\beta_f$', r'$\gamma_f$', r'$\tau_0$', 
              r'$\alpha_\tau$', r'$\beta_\tau$', r'$\gamma_\tau$', r'$\alpha_*$', 
              r'$\beta_*$', r'$\gamma_*$', r'$\epsilon$', r'$e_0$', r'$\rho_0$', r'$z$']
    
    names = ['Phi0', 'PhiI', 'M0', 'alpha0', 'alphaI', 'f0', 
             'alphaf', 'betaf', 'gammaf', 't0', 'alphatau', 
             'betatau', 'gammatau', 'alphastar', 'betastar', 
             'gammastar', 'epsilon', 'e0', 'rho0', 'zval']
    
    # Define bounds for the parameters
    bounds = [(-3.4, -2.4), (-0.6, 0.2), (11, 11.5), (-1.5, -1.), 
              (-0.2, 0.2), (0.01, 0.05), (-0.5, 0.5), (0., 2.), 
              (-0.2, 0.2), (0.1, 10.), (-0.5, 0.5), (-3., 1.), 
              (-0.2, 0.2), (0.9, 1.5), (8., 9.), (-0.5, 0.5), 
              (0.2, 0.5), (0.01, 0.99), (-2., 2.), (0.01, 6.)]


    
    # Set the redshift value and initialize an array for redshifts
    zv = 2.0
    z = np.linspace(0., zv, 10)  # Create an array of redshift values
    
    # Initial parameter values
    initpar = {
        'Phi0': -2.6, 'PhiI': -0.45, 'M0': 11.25, 'alpha0': -1.15, 
        'alphaI': -0.1, 'f0': 0.02, 'alphaf': 0.1, 'betaf': 0.8, 
        'gammaf': 0.1, 't0': 0.8, 'alphatau': -0.1, 'betatau': -2., 
        'gammatau': -0.1, 'alphastar': 1.1, 'betastar': 8.0, 
        'gammastar': 0.1, 'epsilon': 0.3, 'e0': 0.9, 
        'rho0': 0.1, 'zval': zv
    }

    # Set print options for better readability of output
    np.set_printoptions(precision=20)

    
    # Calculate merger rate 
    mrate = mergerrate(M1, q, z, f, **initpar)
    hq = mrate.hmodelt()[0]  # Get the first model output for strain
    print(hq)  # Output the calculated strain values to the console

    # Plotting the results
    plt.figure()
    plt.plot(f, hq)  
    plt.xlabel(r'$\log_{10} (f$ [Hz]$)$', fontsize='x-large')  
    plt.ylabel(r'$\log_{10} h_c$', fontsize='x-large')  
    plt.show()  