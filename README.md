# fall-2025-risk-factors-in-online-courses
This is a repository for Identifying Risk Factors in Online Courses as part of The Erdös Institute Fall 2025 Data Science Bootcamp. 

Team members: James Alex Caramanico, Arina Favilla, James Edward McNally, Feng Zhu

Acknowledgements: We are grateful for the instruction and support of the Erdös Institute, including Alec Clott, Roman Holowinsky, and Stephen Gubkin. We are especially thankful for the generous feedback and support provided by Evelyn Huszar, whose mentorship was crucial. 

#  Table of Contents</p>

1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Model Selection and Performance](#model-selection-and-performance)
4. [Conclusions](#conclusions)
5. [Description of Repository](#description-of-repository)

## Introduction
What early engagement patterns in virtual learning environments predict negative course outcomes? It is well known that performance on assessments and in-class attendance are predictive of final course results. Yet grades often come too late in a class term for early interventions and attendance is difficult, if not impossible, to measure in online learning environments. To address this gap, we developed a model for identifying early risk factors in online courses based on student interaction patterns in a virtual learning environment (VLE). Rather than relying on grades or conventional risk features such as demographics, our model bases its predictions on various facets of student engagement in a VLE. To focus our model on early risk factors, we restricted data to the first three weeks of a course. Our primary key performance indicator was recall–that is, the % of students who failed or withdrew late from the course who were successfully identified by our model–which we maximized while keeping precision above 50% and the false negative rate (i.e., % of failing students that were incorrectly classified as not at-risk) below 20%.

Our analysis will inform instructional approaches and the development of educational technology products to better identify struggling students from metadata early in the course before negative outcomes or obvious risk factors occur. Stakeholders include educational institutions  with an interest in equitable learning outcomes and maximizing student success, as well as educational technology companies (e.g., Blackboard, Canvas) that would be able to improve early warning detection systems. Since our model aims to flag students at risk of failing, it does not predict scores or performance beyond a binary pass / fail outcome. 

## Dataset
We used data from the Open University Learning Analysis Dataset (OULAD), which includes daily logs of student VLE interactions and grades in 7 science and social science online courses occurring in 2013-14. All of the students lived in the UK. Multiple publications have analyzed this dataset, but a systematic review of this literature pointed to feature engineering as an underexplored area for future work. Because most OULAD analyses use data extracted from beyond the course’s first 3 weeks, we found that our model’s focus on the very beginning of a course represents an additional novel approach, as did our lack of reliance on grades as a predictor. 

After creating a new dataset out of the original OULAD, we engineered the following features, all of which pertain only to the pre-course period through the course’s third week:
- Whether a student submitted their first assignment on time, late, or never 
- A student’s total number of interactions with the VLE 
- How regularly a student engaged with the VLE 
- Whether a student’s interactions with the VLE were predominantly focused on content or collaboration
- How diverse a student’s different types of interactions with the VLE were

## Model Selection and Performance

We framed our model as a classification problem focused on predicting negative student performance (i.e., if a student either failed or withdrew from the course after the first 3 weeks). In preliminary modeling, we found that our model performed significantly better as a binary classification problem between students who passed or passed with distinction and students who failed or withdrew from the course after the first 3 weeks. To ensure consistency across different modules, which covered different subjects and had different timelines, we only used data from courses and terms that had an assessment within the first 3 weeks of the term. We did not use the score on the first assessment as part of our model. Our final dataset included 13,476 students across 11 module-presentation combinations, 42.3% of whom either failed or withdrew after the first three weeks. 

After cleaning the data and resolving missing values, we used 5-fold stratified cross-validation to compare (a) Baseline (predicts the global prevalence of the target across the dataset), (b) Logistic Regression, (c) Random Forest, (d) Extra Trees, (e) Gradient Boosting, and (f) Extreme Gradient Boosting. Logistic Regression (0.629 recall, 0.602 F1 score), Random Forest (0.601 recall, 0.592 F1 score), and Extra Trees (0.637 recall, 0.593 F1 score) were the strongest models after hyperparameter tuning.

To maximize recall, we compared the performance of Logistic Regression, Random Forest, and Extra Trees at different classification thresholds (0.35 vs. 0.4 vs. 0.45 vs. 0.5). To determine the ideal threshold, we examined the profile of the “false positive” students–that is, the students who were predicted to fail but in fact went on to pass the class. We decided that we would be comfortable with flagging false positive students if they had at least 1 risk factor early on in the course, which we defined as either (a) having a low score on their initial assessment, (b) having a low total number of VLE interactions, or (c) having turned in their first assessment more than three days late or not at all. As the threshold decreased from 0.5 to 0.35, the percentage of false positive students who met these criteria decreased slightly (from 57.1% to 53.2% for Logistic Regression). We also found that there was a direct relationship between recall and the false positive rate, which represented a significant limit on our model at lower thresholds. Feature importance analysis confirmed the predictive ability of our engineered features.

We settled on a threshold of 0.4, with the understanding that instructors could set different thresholds depending on intervention resources and their tolerance for false positives.. For our final model, we used Logistic Regression, which we felt yielded an ideal balance of KPIs while also being easy to explain to stakeholders. Finally, we ran our tuned model on the test holdout data at a 0.4 threshold, achieving a performance similar to that of the training data.

## Conclusions

In conclusion, our model provides an effective means of predicting at-risk students before conventional risk factors emerge. Future projects could explore the relationship between active learning oriented virtual activities and student success, the effects of retaking courses, and how student-VLE interactions over the course of a term affects retention.  

## Description of Repository

The files are organized as follows:
- `main_pipeline.ipynb` - main Jupyter notebook, used to interact with procedures and scripts throughout the analytic process. This file handles the loading, merging, cleaning, exploratory analysis, and modeling steps of the project. 

Outline of `main_pipeline.ipynb`: 

**Data Preprocessing** (each step under data preprocessing can be run together, or in isolation using previously saved files depending on user need)
  - **Data Merging**: requires `OULAD.zip`, calls functions in `data_prep.py`
  - **EDA: Module-Presentation Comparison**: requires `merged_data.csv` (created in Data Merging), calls functions in `get_course_info.py`
  - **Feature Calculation**: requires `merged_data.csv` (created in Data Merging) and assessments_OULAD.csv (created in Data Merging), calls functions in `feature_calculator.py`
  - **Data Selection**: requires `data_pre_thru_week*.csv` (created in Feature Calculation) and assessments_OULAD.csv (created in Data Merging), calls functions in `data_selection.py`

**Modeling Procedures** (should be run together) 
  - **Define models**: requires `subset_data_pre_thru_week*.csv` (created in Data Selection)
  - **Fit and tune models**: calls functions in `modeler.py`, saves `final_models.pkl` after hyperparameter tuning (since it takes a while to run, allowing for users to skip the hyperparameter tuning step and upload the results in future runs)
  - **Further analysis**: calls functions in `analysis.py`

**Test set performance**: calls functions in `analysis.py` 
     

Python scripts, most of which are called from `main_pipeline.ipynb`:
  - `combine_tables.py` - contains a basic function for ease of combining dataframes necessitated by the analytical pipeline
  - `data_prep.py` - loads the original OULAD dataset (seven separate csvs contained within a zip file downloaded from https://analyse.kmi.open.ac.uk/open-dataset) and merges the data into one csv file after reformatting it from long to wide format. Functions include _unzip_data_ to unzip the original data, _data_merge_ to handle actual merging of data, and _combine_tables_ for ease of merging any additional files later in the pipeline
  - `data_selection.py` - used to refine dataframe to a specified time period, select only modules with assessments within the selected time frame, and remove students with or above max_num_attempts and those who withdrew within the specified time period
  - `false_positive_analysis.py` (and .ipynb) - analyzes the profile of false positives (students who were incorrectly flagged as failing), focusing on how many false positive students have genuine risk factors and the effect of different thresholds on the # and % of false positives in the overall dataset. 
  - `feature_calculator.py` - used to select and calculate desired engineered features for EDA and modeling purposes. Functions include _assessment_features_ to calculate engineered features using assessment data and _vle_features_ to calculate engineered features using VLE interation data . 
  - `feature_engineering_EDA.py` - creates engineered features and performs exploratory data analysis on the relationship between these engineered features in the first week of a course and a student's final grade (final_result)
  - `get_course_info.py` - contains code to extract data about courses for EDA purposes using function _course_meta_info_ 
  - `modeler.py` - Python script with all the modeling code, with options to fit specific models (returning diagnostic metrics) and do hyperparameter tuning (returning tuned models). Includes functions _data_preprocess_, _model_initfit_, and _model_tune_.More specific documentation at top of specific functions in script. 
  - `threshold_analysis.py` - contains functions to analyze effect of varying the classification threshold on various performance metrics.
- `data_handling.png` - image of a flow chart detailing how the OULAD data was merged together into one dataframe, followed by selections of the data and feature calculation. This displays the entire analytical pipeline in as brief a manner as possible
- `final_models.pkl` - saved output of hyperparameter tuning after calling function _model_tune_ for all candidate models 

**NB:** We note that there are a few sources for the original OULAD dataset (the UC Irvine ML Repository, Kaggle, and the Open University [OU] Analyse site). These are mostly similar, but have some minor differences. We emphasize that **the scripts in this branch are written to work with and have been tested on the version from the OU Analyse site** at https://analyse.kmi.open.ac.uk/open-dataset  
