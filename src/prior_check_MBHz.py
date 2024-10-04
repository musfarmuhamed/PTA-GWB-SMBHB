import numpy as np

def schechterf(M, M0, phi0, alpha0):
    """
    Computiing Galaxy stellar mass function using Schechter function 
    """
    return phi0 * np.power(M / M0, 1. + alpha0) * np.exp(-M / M0)

def Mbh(Mbulge, alpha, beta, gamma, zv):
    """
    Calculate the black hole mass based on galaxy bulge mass and redshift.
    """
    return alpha * np.log10(Mbulge / 1.e11) + beta + gamma * zv

def check_p(livepoint):
    """
    Validate the model parameters against prior data.

    Parameters:
    livepoint (dict): Dictionary containing 20 model parameters.

    Returns:
    float: Log likelihood of the model parameters; -inf if invalid.
    """
    # Load prior data
    data = np.genfromtxt('../data/mbulge.dat')
    data0 = np.genfromtxt('../data/numdenrange0.txt')
    data1 = np.genfromtxt('../data/numdenrange1.txt')

    # Define mass and redshift ranges
    m0 = np.logspace(9, 12, 15)
    z0 = [0.4, 0.6, 0.8]
    z1 = [1.5, 2.0, 2.5]
    l = 0.0    

    # Loop through redshift values for initial checks
    for i in z0:
        phi0 = livepoint['Phi0'] + livepoint['PhiI'] * i
        a0 = livepoint['alpha0'] + livepoint['alphaI'] * i
        model = schechterf(m0, 10. ** livepoint['M0'], 10. ** phi0, a0)

        # Check model against prior data
        if (model<=data0[:,6]).all()==True and (model>=data0[:,1]).all()==True:
            for j in z1:
                phi0 = livepoint['Phi0'] + livepoint['PhiI'] * j
                a0 = livepoint['alpha0'] + livepoint['alphaI'] * j
                model = schechterf(m0, 10. ** livepoint['M0'], 10. ** phi0, a0)

                # Validate model against second observational dataset
                if (model<=data1[:,6]).all()==True and (model>=data1[:,1]).all()==True:
                     l += 0.0
                else:
                    return -np.inf
        else:
            return -np.inf

    # Calculate black hole mass for given parameters
    mb = np.logspace(8.9, 11.7, 12)
    model = Mbh(mb, livepoint['alphastar'], livepoint['betastar'], livepoint['gammastar'], livepoint['zval'])
    model2 = Mbh(mb, livepoint['alphastar'], livepoint['betastar'], livepoint['gammastar'], 0)

    # Final validation against the data
    if (model<=data[:,2]).all()==True and (model>=data[:,1]).all()==True:
        if (model2<=data[:,2]).all()==True and (model2>=data[:,1]).all()==True:
            return l
        else:
            return -np.inf
    else:
        return -np.inf
