import numpy as np
import matplotlib.pylab as plt
import mergerrate_MBH_gamma as mr

# Define parameter ranges
M1 = np.linspace(9, 12, 30)      # Mass range for M1
q = np.linspace(0.25, 1, 15)      # Mass ratio range
mc = np.linspace(5, 11, 30)       # Chirp mass  range
mcdiff = (mc[1] - mc[0]) / 2.     # Half the difference in chirp mass 
f = np.linspace(-10, -7, 10)      # Frequency range

# Define parameter names
names =['Phi0', 'PhiI', 'M0', 'alpha0', 'alphaI', 
        'f0' , 'alphaf', 'betaf', 'gammaf' , 
        't0' , 'alphatau', 'betatau' , 'gammatau', 
        'alphastar', 'betastar', 'gammastar' ,'epsilon', 
        'e0', 'rho0' , 'zval']

# Load chain data
data = np.loadtxt('../output/chain_1.txt')
chaint = data[:, :-4]

# throw out first 25% of chain as burn in 
burn = int(0.25*chaint.shape[0])

# Calculate Mbulge based on the defined mass range
Mbulg = np.array([10.**M1[j] * mr.mfraction(M1[j]) for j in range(len(M1))])

# Initialize arrays to store results
arr_hc, arr_dndz, arr_dndm = [], [], []     # Array for hc values, dN/dz values, dN/dM values 
arr_mbh, arr_mbulge, arr_z, arr_mc = [], [], [], [] # Array for black hole masses, bulge masses, redshifts, chirp masses 

# Number of iterations for random sampling
n=10000  

# Loop to sample from the chain and perform calculations
for i in range(n):
    ran = np.random.randint(len(chaint[burn:, :]))  # Randomly select an index
    initpar = dict(zip(names, chaint[burn + ran, :]))  # Initialize parameters
    zv = initpar['zval']  # Redshift value
    z = np.linspace(0., zv, 10)  # Redshift range
    mrate = mr.mergerrate(M1, q, z, f, **initpar)  # Calculate merger rate

    # Plot hc values
    out = mrate.hmodelt()
    fig1 = plt.figure(1)
    plt.plot(f, out[0])
    plt.xlabel(r'$\log_{10} (f$ [Hz]$)$', fontsize='x-large')
    plt.ylabel(r'$\log_{10} h_c$', fontsize='x-large')       
    arr_hc.append(out[0])
    
    # Calculate and plot dN/dM
    d2n = np.sum(out[1], axis=1)
    dn = np.multiply(d2n, mrate.zpdiff / mcdiff)
    dndm = np.sum(dn, axis=1)
    fig2 = plt.figure(2)
    plt.plot(mc, np.log10(dndm))
    plt.xlabel(r'$\log_{10} \mathcal{M} (M_\odot)$', fontsize='x-large')
    plt.ylabel(r'$\log_{10} \frac{dN}{dVd\log  \mathcal{M}}(Mpc^{-3})$', fontsize='x-large')       
    arr_dndm.append(dndm)
    arr_mc.append(mc)
    
    # Calculate and plot dN/dz
    dndz = np.sum(d2n,axis = 0)
    fig3 = plt.figure(3)
    plt.plot(z,np.log10(dndz))
    plt.xlabel(r'$z$', fontsize='x-large')
    plt.ylabel(r'$\log_{10} \frac{dN}{dVdz}(Mpc^{-3})$', fontsize='x-large')       
    arr_dndz.append(dndz)
    arr_z.append(z)

    # Calculate and plot black hole masses
    MBH = mrate.MBH(Mbulg,zv)
    fig4 = plt.figure(4)
    plt.plot(np.log10(Mbulg),MBH)
    plt.xlabel(r'$\log_{10} M_{bulge} (M_\odot)$', fontsize='x-large')
    plt.ylabel(r'$\log_{10} M_{BH} (M_\odot)$', fontsize='x-large')       
    arr_mbulge.append(Mbulg)
    arr_mbh.append(MBH)

    # Print progress every 100 iterations
    if (i + 1) % 100 == 0:
        print("Percentage completed: ", (i + 1) / n * 100)

# Save figures
fig1.savefig('../output/plot_hc.png',bbox_inches='tight')
fig2.savefig('../output/plot_dn_dm.png',bbox_inches='tight')
fig3.savefig('../output/plot_dn_dz.png',bbox_inches='tight')
fig4.savefig('../output/plot_MBH.png',bbox_inches='tight')
#fig1.savefig('../output/plot_hc.pdf',bbox_inches='tight')
#fig2.savefig('../output/plot_dn_dm.pdf',bbox_inches='tight')
#fig3.savefig('../output/plot_dn_dz.pdf',bbox_inches='tight')
#fig4.savefig('../output/plot_MBH.pdf',bbox_inches='tight')

# Save results to .npy files
np.save('../output/hc_value', arr_hc)
np.save('../output/fhc', f)
np.save('../output/dndz', arr_dndz)
np.save('../output/dndmc', arr_dndm)
np.save('../output/mbh_value', arr_mbh)
np.save('../output/mbh_value', arr_mbulge)
np.save('../output/zdndz', arr_z)
np.save('../output/mcdndmc', arr_mc)

# Uncomment below to display plots if desired
#plt.show()