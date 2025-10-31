# fall-2025-risk-factors-in-online-courses
This branch is home to the primary interface of our project. The files are organized as follows:
- `main_pipeline.ipynb` - main Jupyter notebook, used to interact with procedures and scripts throughout the analytic process. This file handles the loading, merging, cleaning, exploratory analysis, and modeling steps of the project. Each step can be run together, or in isolation depending on user need
- Python scripts, most of which are called from `main_pipeline.ipynb`:
  - `combine_tables.py` - contains a basic function for ease of combining dataframes necessitated by the analytical pipeline
  - `data_prep.py` - loads the original OULAD dataset (seven separate csvs contained within a zip file downloaded from https://analyse.kmi.open.ac.uk/open-dataset) and merges the data into one csv file after reformatting it from long to wide format. Functions include _unzip_data_ to unzip the original data, _data_merge_ to handle actual merging of data), and _combine_tables_ for ease of merging any additional files later in the pipeline
  - `data_selection.py` - used to refine dataframe to a specified time period, select only modules with assessments within the selected time frame, and remove students with or above max_num_attempts and those who withdrew within the specified time period
  - `false_positive_analysis.py` (and .ipynb, if we don't combine these in with other files) - analyzes the profile of false positives (students who were incorrectly flagged as failing), focusing on how many false positive students have genuine risk factors and the effect of different thresholds on the # and % of false positives in the overall dataset. 
  - `feature_calculator.py` - used to select and calculate desired engineered features for EDA and modeling purposes
  - `feature_engineering_EDA.py` - creates engineered features and performs exploratory data analysis on the relationship between these engineered features in the first week of a course and a student's final grade (final_result)
  - `get_course_info.py` - contains code to extract data about courses for EDA purposes.
  - `modeler.py` - Python script with all the modeling code, with options to fit specific models (returning diagnostic metrics) and do hyperparameter tuning (returning tuned models). More specific documentation at top of specific functions in script.
  - `threshold_analysis.py` - contains functions to analyze effect of varying the classification threshold on various performance metrics.
- `data_handling.png` - image of a flow chart detailing how the OULAD data was merged together into one dataframe, followed by selections of the data and feature calculation. This essentially displays the entire analytical pipeline in as brief a manner as possible
- `final_models.pkl` - ***ARINA***

We note that there are a few sources for the original OULAD dataset (the UC Irvine ML Repository, Kaggle, and the Open University [OU] Analyse site). These are mostly similar, but have some minor differences. We emphasize that the scripts in this branch are written to work with and have been tested on the version from the OU Analyse site at https://analyse.kmi.open.ac.uk/open-dataset  
