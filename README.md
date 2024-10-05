# PTA-GWB-SMBHB: Mass-Redshift Dependency of SMBHB for the Gravitational Wave Background
*M. M. Kozhikkal, S. Chen, G. Theureau, M. Habouzit, A. Sesana*  
Published in: **MNRAS, 2024**  

[![DOI](https://img.shields.io/badge/DOI-10.1093/mnras/stae1219-blue)](https://doi.org/10.1093/mnras/stae1219)


This repository contains the code and data used for the analysis presented in the paper:
**"Mass-redshift dependency of supermassive black hole binaries for the gravitational wave background"**,  
published in *Monthly Notices of the Royal Astronomical Society (MNRAS)*, 531, 1931–1950, 2024.

The code explores the co-evolution of supermassive black holes (SMBHs) and their host galaxies, their mass-redshift dependencies, and the impact on the stochastic gravitational wave background (GWB) from pulsar timing arrays (PTA).


## Repository Structure

```bash
├── src/                       # Core scripts for the analysis
│   ├── BH_bulge_mass_plots.py
│   ├── mergerrate_MBH_gamma.py
│   ├── prior_check_MBHz.py
│   ├── bayesian_analysis_mcmc.py
│   ├── corner_plot.py
│   └── merger_rate_analysis.py
├── data/                      # Data files used for analysis
│   ├── data_scaling_relation_allsimulations_Habouzit21_error.txt
│   ├── numdenrange0.txt
│   ├── numdenrange1.txt
│   ├── mbulge.dat
│   └── input_hc.dat
├── output/                    # Output from the scripts
│   ├── chain_1.txt
│   ├── jumps.txt
│   ├── DEJump_jump.txt
│   ├── covarianceJumpProposalAM_jump.txt
│   ├── result1.txt
│   ├── covarianceJumpProposalSCAM_jump.txt
│   ├── cov.npy
│   ├── dndmc.npy
│   ├── dndz.npy
│   ├── fhc.npy
│   ├── hc_value.npy
│   ├── mcdndmc.npy
│   └── zdndz.npy
├── README.md                  # Project overview (this file)
└── requirements.txt           # Python dependencies
```

## Overview

The code is designed to model the stochastic gravitational wave background (GWB) produced by supermassive black hole binaries (SMBHBs) and test it against GWB from pulsar timing arrays (PTA). The analysis uses data from large-scale cosmological simulations, Bayesian analysis, and posterior sampling.

The key goals of the project are:
- To analyze the redshift-dependent evolution of the SMBH–bulge mass relation.
- To compute the GWB intensity as a function of frequency.
- To use Bayesian inference to constrain model parameters from simulated PTA data.

### Features:
1. **BH-bulge Mass Relation with Redshift** (`src/BH_bulge_mass_plots.py`): 
   - Analyzes the evolution of the SMBH–galaxy bulge mass relation with redshift using data from EAGLE, Illustris, TNG100, TNG300, Horizon-AGN, and SIMBA simulations.
   
2. **Astrophysical Model for GWB** (`src/mergerrate_MBH_gamma.py`): 
   - Constructs a parametric model to compute the GWB from SMBHBs, incorporating the galaxy stellar mass function (GSMF), pair fraction, merger time-scale, and binary eccentricity.
   
3. **Bayesian Analysis** (`src/bayesian_analysis_mcmc.py`, `src/prior_check_MBHz.py`): 
   - Performs Bayesian inference using Markov Chain Monte Carlo (MCMC) to analyze the parameter space of the GWB model and generates output chain files for post-processing.

4. **Corner Plot, Merger Rate and GWB Strain Analysis** (`src/corner_plot.py`, `src/merger_rate_analysis.py`): 
   - Computes and plots the corner plot, SMBHB merger rates and characteristic strain values based on the Bayesian posterior distributions.

## Installation

To set up the environment, install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

1. **Analyze BH–Bulge Mass Relations**:
   Run the script to generate mass and redshift-dependent plots of the BH–bulge mass relation:

   ```bash
   python src/BH_bulge_mass_plots.py
   ```

2. **Compute Merger Rates and GWB Strain**:
   Use the merger rate analysis script to compute and visualize the GWB strain for the given parameters:

   ```bash
   python src/merger_rate_analysis.py
   ```

3. **Run Bayesian Analysis**:
   Execute the Bayesian MCMC analysis for posterior sampling:

   ```bash
   python src/bayesian_analysis_mcmc.py
   ```

4. **Plot Corner Plot**:
   Visualize the posterior distributions of the Bayesian analysis:

   ```bash
   python src/corner_plot.py
   ```

5. **Merger Rate and GWB Strain Analysis**:
   Visualize the merger rate and GWB strain distributions of the Bayesian analysis:

   ```bash
   python src/merger_rate_analysis.py
   ```

## Data Files

- **BH Mass**: `data/data_scaling_relation_allsimulations_Habouzit21_error.txt` provides the scaling relations for BH mass and galaxy mass from six cosmological simulations.
- **Priors**: `data/numdenrange0.txt` and `data/numdenrange1.txt` contain the prior data used in Bayesian analysis.
- **GWB Detection**: `data/input_hc.dat` provides the simulated PTA data for GWB detection.

## Outputs

The outputs of the analysis include MCMC chains, jump proposals, covariance matrices, merger rates, and strain values. These are stored in the `output/` folder.

## Citation

If you use this code in your research, please cite the following publication:

> Musfar Muhamed Kozhikkal, Siyuan Chen, Gilles Theureau, Mélanie Habouzit, Alberto Sesana,  
> *"Mass-redshift dependency of supermassive black hole binaries for the gravitational wave background,"*  
> MNRAS, 531, 1931–1950 2024. DOI: [10.1093/mnras/stae1219](https://doi.org/10.1093/mnras/stae1219)

```bibtex
@article{kozhikkal2024,
       author = {{Kozhikkal}, Musfar Muhamed and {Chen}, Siyuan and {Theureau}, Gilles and {Habouzit}, M{\'e}lanie and {Sesana}, Alberto},
        title = "{Mass-redshift dependency of supermassive black hole binaries for the gravitational wave background}",
      journal = {MNRAS},
         year = 2024,
        month = jun,
       volume = {531},
       number = {1},
        pages = {1931-1950},
          doi = {10.1093/mnras/stae1219},
archivePrefix = {arXiv},
       eprint = {2305.18293},
 primaryClass = {astro-ph.CO}
}
```
