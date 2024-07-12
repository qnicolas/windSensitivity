# Description
This folder contains data and code to reproduce figures appearing in Nicolas & Boos (2024, submitted) - *Sensitivity of tropical orographic precipitation to wind speed with implications for future projections*. 

tools.py contains several averaging, linear regressions, statistics, etc. utilities.
wave_precip_models.py contains linear mountain wave and orographic precipitation models (and one nonlinear model).
makeFigures.ipynb leverages all of the above to produce the figures.

# Running the code
The first step is to retrieve the data, accessible at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11479598.svg)](https://doi.org/10.5281/zenodo.11479598). Create a 'data' folder and place the four subfolders (wrfData, cmipData, regionsData and globalData) in it.

A .yml file is included that contains all necessary python packages to run the code and produce the figures. Create a conda environment using 'conda env create -f environment.yml' (switch conda to mamba for a faster installation), then activate with conda activate orogconv, open makeFigures.ipynb in jupyter and you are hopefully all set!

# Contact
For any questions, contact qnicolas --at-- berkeley --dot-- edu
