# fall-2025-risk-factors-in-online-courses
This branch is home to the primary interface of our project. The files are organized as follows:
- `main_pipeline.ipynb` - main Jupyter notebook, used to interact with procedures and scripts throughout the analytic process. This file handles the loading, merging, cleaning, exploratory analysis, and modeling steps of the project. Each step can be run together, or in isolation depending on user need
- Python scripts:
  - `data_prep.py` - this file loads the original OULAD dataset (seven separate csvs contained within a zip file that was downloaded from their website - https://analyse.kmi.open.ac.uk/open-dataset) and merges the data into one csv file after reformatting it from long to wide format. Functions include _unzip_data_ to unzip the original data, _data_merge_ to handle actual merging of data), and _combine_tables_ for ease of merging any additional files later in the pipeline
  - `feature_calculator.py` - python script, called in `main_pipeline.ipynb`, used to select and calculate desired engineered features for EDA and modeling purposes
  - `feature_engineering_EDA.py` - **
  - `modeler.py` - python script with all the modeling code, with options to fit specific models and choose certain values of hyperparameters. returns/prints model fit results, diagnostics **
- `data_merging.png` - image of a flow chart detailing how the OULAD data was merged together into one dataframe 
