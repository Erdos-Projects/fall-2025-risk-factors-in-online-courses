# fall-2025-risk-factors-in-online-courses
This branch is home to the primary interface of our project. The files are organized as follows:
data_merging.png - image of a flow chart detailing how the OULAD data was merged together into one file
data_merger.py - python script, called in "main_pipeline" when necessaey to execute the merging of the raw OULAD data, after downloading the zip file from their website
data_prep.py - this file completes data cleaning and management, resolving issues we noticed with the raw data **
feature_calculator.py - python script, called in "main_pipeline", used to select and calculate desired engineered features for EDA and modeling purposes
feature_engineering_EDA.py - **
main_pipeline.ipynb - python notebook script, used to interact with procedures and scripts throughout the analytic process. this file handles the loading, merging, cleaning, exploratory analysis, and modeling steps of the project. each can be run together, or in isolation depending on user need
modeler.py - python script with all the modeling code, with options to fit specific models and choose certain values of hyperparameters. returns/prints model fit results, diagnostics **
