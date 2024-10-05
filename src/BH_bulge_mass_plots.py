import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# List of simulation names 
simulations = ['EAGLE', 'Illustris', 'TNG100', 'Horizon-AGN', 'SIMBA', 'TNG300']

# Load the data from the BH mass data of simulations file
data = np.loadtxt('../data/data_scaling_relation_allsimulations_Habouzit21_error.txt')


# Create a DataFrame from the loaded data
df_all = pd.DataFrame(data=data, columns=["simulation", "redshift", "Mstellar", "Mbh", "Mbh15p", "Mbh85p"])

# Convert simulation identifiers from float to string and map to their respective names
df_all['simulation'] = df_all['simulation'].astype(str)
df_all['simulation'] = df_all['simulation'].replace({
    '0.0': 'Illustris',
    '1.0': 'TNG100',
    '3.0': 'Horizon-AGN',
    '2.0': 'TNG300',
    '4.0': 'EAGLE',
    '5.0': 'SIMBA'
})

# Convert redshift from float to integer
df_all['redshift'] = df_all['redshift'].astype(int)

# Define the bulge mass calculation based on stellar mass
def calculate_bulge_mass(mass):
    """Calculate bulge mass based on stellar mass."""
    if mass > 10.:
        return (np.sqrt(6.9) * np.exp(-6.9 / (2. * (mass - 10.))) / (mass - 10.)**1.5 + 0.615) * 10**mass
    else:
        return 0.615 * 10**mass

# Apply the bulge mass calculation to the DataFrame
df_all['Mbulge'] = np.log10(df_all['Mstellar'].apply(calculate_bulge_mass))

# Create a mapping DataFrame for simulation indices
df_mapping = pd.DataFrame({'simulation': simulations})
df_mapping = df_mapping.reset_index().set_index('simulation')

# Map the numerical index of simulations to the DataFrame
df_all['simulation_num'] = df_all['simulation'].map(df_mapping['index'])
df_all.sort_values(by=['simulation_num', 'redshift', 'Mstellar'], inplace=True)

# Calculate the modified bulge mass
df_all['Mb_M0'] = df_all['Mbulge'] - 11

# Calculate the error in black hole mass
df_all['Mbh-er'] = df_all['Mbh'] - df_all['Mbh15p']
df_all['Mbh+er'] = df_all['Mbh85p'] - df_all['Mbh']

# Select relevant columns for further analysis
df = df_all[["simulation", "redshift", "Mb_M0", "Mbh", "Mbh-er", "Mbh+er"]]

# Define colors for each simulation
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown']

# Load bulge mass limit data
mbulge_limit = np.genfromtxt('../data/mbulge.dat')

# Scatter plot of bulge mass vs black hole mass
for index, sim in enumerate(simulations):
    df_sim = df_all[df_all['simulation'] == sim]
    plt.scatter(df_sim['Mbulge'], df_sim['Mbh'], label=sim, color=colors[index])

plt.legend()
plt.plot(mbulge_limit[:, 0], mbulge_limit[:, 1], 'dimgray', linestyle='--', label='mbulge limit 1')
plt.plot(mbulge_limit[:, 0], mbulge_limit[:, 2], 'dimgray', linestyle='--', label='mbulge limit 2')

plt.xlabel(r'$\log_{10}M_{bulge}$')
plt.ylabel(r'$\log_{10}M_{BH}$')
plt.show()

# Define linear fit function
def linear_fit(x, m, c):
    """Linear fit function."""
    return m * x + c

def linesl(x, c):
    """Linear fit function with given slope."""
    return popts[0] * x + c

def plot_with_error(data):
    """Create subplots for each simulation and redshift, including error bars."""
    fig, axs = plt.subplots(6, 6, figsize=(30, 36), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.2)
    
    global gpara_simr
    gpara_simr = [[]]
    print(r"Simulation, z, $\alpha_*$, $\alpha_*$ error, $\beta_*$, $\beta_*$ error, $\gamma_*$, $\epsilon$")

    for i, sim in enumerate(simulations):
        df_sim = data[data['simulation'] == sim]
        yerr_sim = (df_sim['Mbh-er'] + df_sim['Mbh+er']) / 2.
        
        # Fit a line to the data
        global popts
        popts, pcovs = curve_fit(linear_fit, df_sim['Mb_M0'], df_sim['Mbh'], sigma=yerr_sim)
        
        for z in range(6):
            df_sim_z = df_sim[df_sim['redshift'] == z]

            axs[i,z].errorbar(df_sim_z['Mb_M0']+11,df_sim_z['Mbh'],yerr=[df_sim_z['Mbh-er'],df_sim_z['Mbh+er']]
                             , fmt='o',color=colors[i], ecolor='darkgray', elinewidth=4, capsize=1)

            yerr=(df_sim_z['Mbh-er']+df_sim_z['Mbh+er'])/2.
            
            poptsz, pcovsz = curve_fit(linesl, df_sim_z['Mb_M0'], df_sim_z['Mbh'],sigma=yerr)

            if z==0:
                mstar = poptsz[0]
                betas_sl = 0
            else:
                betas_sl = (poptsz[0]-mstar)/z
            scatters = (df_sim_z['Mbh']-linesl(df_sim_z['Mb_M0'],*poptsz)).abs().sum()/len(df_sim_z)
            axs[i,z].plot(df_sim_z['Mb_M0']+11,linesl(df_sim_z['Mb_M0'],*poptsz), color=colors[i])
            axs[i,z].text(.05, .93, r'$\alpha_*$=%.2f, $\beta_*$=%.2f'
                          %(popts[0],mstar),transform=axs[i,z].transAxes, fontsize=20)
            axs[i,z].text(.05, .85, r'$\gamma_*$=%.2f, $\epsilon$=%.2f'
                          %(betas_sl,scatters),transform=axs[i,z].transAxes, fontsize=20)
            axs[i,z].set_title(' %s - z = %d'%(sim,z), fontsize=26)
            axs[i,z].set_xlabel(r'$\log  M_{bulge} $', fontsize=28)
            axs[i,z].set_ylabel(r'$\log M_{BH}$', fontsize=28)
            axs[i,z].tick_params(axis='x', labelsize=13)
            axs[i,z].tick_params(axis='y', labelsize=18)
            axs[i,z].label_outer()

            gpara_simr.append([z,simulations[i],popts[0],mstar,betas_sl,scatters])

            print(sim,z, '%.2f  %.3f %.2f %.3f %.2f %.2f'
                          %(popts[0],pcovs[0,0]**0.5,mstar,pcovsz[0,0]**0.5,betas_sl,scatters))
            
    plt.show()

plot_with_error(df)

# DataFrame for storing fitting parameters
gpara_df = pd.DataFrame(gpara_simr[1:],columns=['z','simulation','alphastar','betastar','gammastar','epsilon'])


parar= [r'$\alpha_*$',r'$\beta_*$',r'$\gamma_*$',r'$\epsilon$']

def plot_parameters(data):
    """Plot the variation of fitting parameters with redshift."""
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    plt.subplots_adjust(wspace=0.2)
    
    for index, sim in enumerate(simulations):
        for i in range(4):
            data_sim = data[data['simulation'] == sim]
            if i == 2:
                data_sim = data_sim[data_sim['z'] != 0]
                
            if i>1:
                ax[i-2].plot(data_sim['z'], data_sim.iloc[:, 2 + i], label=sim, color=colors[index])
                ax[i-2].scatter(data_sim['z'], data_sim.iloc[:, 2 + i], color=colors[index])
                ax[i-2].set_xlabel(r'Redshift z', fontsize=20)
                ax[i-2].set_ylabel('%s' % (parar[i]), fontsize=20)
                ax[i-2].legend()

    plt.show()
    
plot_parameters(gpara_df)
