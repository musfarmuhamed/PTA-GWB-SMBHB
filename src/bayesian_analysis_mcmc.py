#!/usr/bin/env python

import numpy as np
import mergerrate_MBH_gamma as mr
import prior_check_MBHz
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

# Define parameter bounds for the model
"""
names =['Phi0', 'PhiI', 'M0', 'alpha0', 'alphaI', 
        'f0' , 'alphaf', 'betaf', 'gammaf' , 
        't0' , 'alphatau', 'betatau' , 'gammatau', 
        'alphastar', 'betastar', 'gammastar' ,'epsilon', 
        'e0', 'rho0' , 'zval']
"""
bounds = [
    (-3.4, -2.4), (-0.6, 0.2), (11, 11.5),  (-1.5, -1.0), (-0.2, 0.2), 
    (0.01, 0.05), (-0.5, 0.5), (0.0, 2.0),  (-0.2, 0.2), 
    (0.1, 10.0),  (-0.5, 0.5), (-3.0, 1.0), (-0.2, 0.2), 
    (0.8, 1.9),   (7.9, 9.2),  (-0.5, 0.5), (0.05, 0.5), 
    (0.01, 0.99), (-2.0, 2.0), (0.1, 5.0)
]

# Load observational data
data = np.genfromtxt('../data/input_hc.dat')

def in_bounds(par):
    """
    Check if the parameters are within the defined bounds.
    """
    return all(bounds[i][0] < par[i] < bounds[i][1] for i in range(len(par)))

def log_prior(par):
    """
    Check with the prior of the parameters.
    """
    if in_bounds(par):
        param = dict(
            Phi0=par[0], PhiI=par[1], M0=par[2], alpha0=par[3], alphaI=par[4], 
            f0=par[5], alphaf=par[6], betaf=par[7], gammaf=par[8], 
            t0=par[9], alphatau=par[10], betatau=par[11], gammatau=par[12], 
            alphastar=par[13], betastar=par[14], gammastar=par[15], epsilon=par[16], 
            e0=par[17], rho0=par[18], zval=par[19]
        )
        return prior_check_MBHz.check_p(param)
    else:
        return -np.inf

def log_likelihood(par):
    """
    Calculate the log likelihood of the model given the parameters.
    """
    # Define mass, mass ratio and frequency ranges
    M1 = np.linspace(9, 12, 25)
    q = np.linspace(0.25, 1, 10)
    f = data[:, 0]

    # Prepare parameter dictionary for the model
    initpar = dict(
        Phi0=par[0], PhiI=par[1], M0=par[2], alpha0=par[3], alphaI=par[4], 
        f0=par[5], alphaf=par[6], betaf=par[7], gammaf=par[8], 
        t0=par[9], alphatau=par[10], betatau=par[11], gammatau=par[12], 
        alphastar=par[13], betastar=par[14], gammastar=par[15], epsilon=par[16], 
        e0=par[17], rho0=par[18], zval=par[19]
    )

    z = np.linspace(0., par[19], 15)
    model = mr.mergerrate(M1, q, z, f, **initpar).hmodelt(fbin=None)[0]
    
    # Calculate likelihood
    likelihood = -0.5 / (data[:, 2] ** 2) * (model - data[:, 1]) ** 2 - 0.5 * np.log10(2 * np.pi)
    return np.sum(likelihood)

# Initial parameter values
x0 = np.array([-2.8, -0.2, 11.25, -1.25, 0.0, 
               0.025, 0.0, 0.8, 0.0, 
               1.0, 0.0, -0.5, 0.0, 
               1.0, 8.5, 0.1, 0.4, 
               0.5, 0.0, 1.5])

ndim = len(x0)

# Number of samples to draw
N = 500000

# Load covariance matrix
# cov = np.load('../output/cov.npy')
cov = np.diag(np.ones(ndim) * 0.01**2)

# Define parameter groups for sampling
groups = [
    list(range(ndim)),
    [0, 1, 2, 3, 4],  # 5 GSMF parameters
    [5, 6, 7, 8],     # 4 pair fraction parameters
    [9, 10, 11, 12],  # 4 merger time parameters
    [13, 14, 15, 16], # Mbulge MBH parameters
    [17, 18],         # ecc + rho0
    [19]              # z
]

# Initialize and run the sampler
sampler = ptmcmc(ndim, log_likelihood, log_prior, cov, outDir='./../output', resume=True, groups=groups)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
