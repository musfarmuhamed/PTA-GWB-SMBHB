import corner
import numpy as np

# Define parameter names for the corner plot
parameter_names = [
    'Phi0', 'PhiI', 'M0', 'alpha0', 'alphaI', 
    'f0', 'alphaf', 'betaf', 'gammaf', 
    't0', 'alphatau', 'betatau', 'gammatau', 
    'alphastar', 'betastar', 'gammastar', 'epsilon', 
    'e0', 'rho0', 'zval'
]

# LaTeX-friendly parameter names for display
parameter_names_latex = [
    r'$\Phi_0$', r'$\Phi_I$', r'$M_0$', r'$\alpha_0$', r'$\alpha_I$', 
    r'$f_0$', r'$\alpha_f$', r'$\beta_f$', r'$\gamma_f$', 
    r'$\tau_0$', r'$\alpha_\tau$', r'$\beta_\tau$', r'$\gamma_\tau$', 
    r'$\alpha_*$', r'$\beta_*$', r'$\gamma_*$', r'$\epsilon$', 
    r'$e_0$', r'$\rho_0$', r'$z_m$'
]

# Load the MCMC chain data from a text file
data = np.loadtxt('../output/chain_1.txt')

# Exclude the last four columns which are not needed for plotting
chaint = data[:, :-4]

# Determine the number of samples to burn (25% of the total)
burn = int(0.25 * chaint.shape[0])

# Generate the corner plot using the remaining samples
figure = corner.corner(
    chaint[burn:, :], 
    labels=parameter_names_latex, 
    show_titles=True, 
    quantiles=[0.159, 0.5, 0.841]
)

# Save the corner plot to a PDF file
figure.savefig('../output/cornerplot.pdf', bbox_inches='tight')

# Uncomment below to display plot if desired
# plt.show()