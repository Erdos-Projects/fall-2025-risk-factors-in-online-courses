# fall-2025-risk-factors-in-online-courses
This branch is home to the primary interface of our project. The files are organized as follows:
- `main_pipeline.ipynb` - main Jupyter notebook, used to interact with procedures and scripts throughout the analytic process. This file handles the loading, merging, cleaning, exploratory analysis, and modeling steps of the project. Each step can be run together, or in isolation depending on user need
- Python scripts:
  - `analysis.py` - *****FENG/JAMES  
  - `combine_tables.py` - contains a basic function for ease of combining dataframes necessitated by the analytical pipeline
  - `data_prep.py` - this file loads the original OULAD dataset (seven separate csvs contained within a zip file that was downloaded from their website - https://analyse.kmi.open.ac.uk/open-dataset) and merges the data into one csv file after reformatting it from long to wide format. Functions include _unzip_data_ to unzip the original data, _data_merge_ to handle actual merging of data), and _combine_tables_ for ease of merging any additional files later in the pipeline
  - `data_selection.py` - python script, called in `main_pipeline.ipynb`, used to refine dataframe to a specified time period, select only modules with assessments within the selected time frame, and remove students with or above max_num_attempts and those who withdrew within the specified time period
  - `feature_calculator.py` - python script, called in `main_pipeline.ipynb`, used to select and calculate desired engineered features for EDA and modeling purposes
  - `feature_engineering_EDA.py` - *****JAMES
  - `get_course_info.py` *****ARINA
  - `modeler.py` - python script with all the modeling code, with options to fit specific models and choose certain values of hyperparameters. returns/prints model fit results, diagnostics *****FENG/JAMES
- `data handling.png` - image of a flow chart detailing how the OULAD data was merged together into one dataframe, followed by selections of the data and feature calculation. This essentially displays the entire analytical pipeline in as brief a manner as possible
