def feature_calculator(df,features=['submission','total','focus','regularity','diversity','demographics'],
                       n_weeks=3,write_to=None):
    # parameters: 
    # df is the merged dataframe containing all necessary variables
    # features is a list of feature types we wish to calculate. all are included by default
    # n_weeks determines the time period to be selected from the beginning of the data set
    # write_to can be used to specify a file path if the user wishes to save the final dataframe in a csv
    
    import numpy as np
    import pandas as pd

    features = [item.lower() for item in features] # avoid case-sensitive errors

    # Create a "week" column relative to each code_presentation's starting date.
    # Dates 0.0 - 6.0 are week 1, 7.0 - 13.0 are week 2, etc. 
    # The weeks *before* date 0.0 are assigned to negative values.
    df['week'] = df.groupby('code_presentation')['date'].transform( # create "week" column
        lambda x: ((x // 7) + 1).where(x >= 0, x // 7))
    
    df_pre_w3 = df[df['week'] <= n_weeks] # Filter to pre-course through week n and sum interactions per student

    student_level_columns = [] # empty list of student-level characteristics to be appended

    if 'submission' in features:
        ## Add submission related features for FIRST ASSIGNMENT only
        # - Relative submission date
        # - Submission type (late vs. early)

        # Find the first assignment for each student (by minimum due_date)
        first_assessment = df[df['id_assessment'].notna()].groupby(['id_student', 'code_presentation'])['due_date'].min().reset_index()
        first_assessment = first_assessment.rename(columns={'due_date': 'first_due_date'})

        # Merge first assessment due date back to main dataframe
        df = df.merge(first_assessment, on=['id_student', 'code_presentation'], how='left')

        # For rows where the student submitted the first assessment (where due_date matches first_due_date and id_assessment exists)
        # Calculate relative submission date and type
        df_first_submission = df[(df['id_assessment'].notna()) & (df['due_date'] == df['first_due_date'])].copy()
        df_first_submission['relative_submission_date'] = df_first_submission['first_due_date'] - df_first_submission['date']
        df_first_submission['submission_type'] = np.where(df_first_submission['relative_submission_date'] < 0, "Late", "Early")

        # Get one row per student with their first submission metrics
        first_submission_metrics = df_first_submission.groupby(['id_student', 'code_presentation']).agg({
            'relative_submission_date': 'first',  # Take the submission date (should be one row per student for first assessment)
            'submission_type': 'first'
        }).reset_index()

        # Merge back to main dataframe
        df = df.merge(first_submission_metrics, on=['id_student', 'code_presentation'], how='left', suffixes=('', '_first'))

        # Clean up: if we created duplicate columns, keep the new ones
        if 'relative_submission_date_first' in df.columns:
            df['relative_submission_date'] = df['relative_submission_date_first']
            df = df.drop(columns=['relative_submission_date_first'])
        if 'submission_type_first' in df.columns:
            df['submission_type'] = df['submission_type_first']
            df = df.drop(columns=['submission_type_first'])

        # Drop the helper column
        df = df.drop(columns=['first_due_date'])

        student_level_columns.extend(['relative_submission_date','submission_type'])

    if 'total' in features:
        ## Add feature that sums total VLE interactions through week 3
        vle_columns = ['quiz', 'questionnaire', 'externalquiz', 'oucontent', 'page', 'resource', 'url', 'homepage', 
                    'glossary', 'subpage', 'folder', 'forumng', 'oucollaborate', 'ouelluminate', 'ouwiki', 'sharedsubpage', 
                    'dataplus', 'repeatactivity', 'dualpane', 'htmlactivity']

        total_interactions_pre_w3 = df_pre_w3.groupby(['id_student', 'code_presentation'])[vle_columns].sum().sum(axis=1)
        total_interactions_pre_w3 = total_interactions_pre_w3.rename('total_vle_interactions_w3')
        df = df.merge(total_interactions_pre_w3, on=['id_student', 'code_presentation'], how='left')
        student_level_columns.extend('total_vle_interactions_w3')

    if 'focus' in features:
        ## Add VLE interaction category features:
        # - Content type interaction (percentage)
        # - Collaborative type interaction (percentage)

        # This creates 2 columns for pre-course through week 3:
        # - collaborative_focus_pre_w3
        # - content_focus_pre_w3
        content_types = ['oucontent', 'page', 'resource', 'url', 'homepage', 'glossary', 'subpage', 'folder']
        collaborative_types = ['forumng', 'oucollaborate', 'ouelluminate', 'ouwiki', 'sharedsubpage']

        student_totals = df_pre_w3.groupby(['id_student', 'code_presentation'])[vle_columns].sum()

        content_focus_pre_w3 = student_totals[content_types].sum(axis=1) / student_totals[vle_columns].sum(axis=1)
        content_focus_pre_w3 = content_focus_pre_w3.fillna(0).rename("content_focus_pre_w3")

        collaborative_focus_pre_w3 = student_totals[collaborative_types].sum(axis=1) / student_totals[vle_columns].sum(axis=1)
        collaborative_focus_pre_w3 = collaborative_focus_pre_w3.fillna(0).rename("collaborative_focus_pre_w3")

        df = df.merge(content_focus_pre_w3, on=['id_student', 'code_presentation'], how='left')
        df = df.merge(collaborative_focus_pre_w3, on=['id_student', 'code_presentation'], how='left')
        student_level_columns.extend(['content_focus_pre_w3','collaborative_focus_pre_w3'])

    if 'regularity' in features:
        ## Add regularity features for pre-course through week 3:
        # - Active days per week
        # - Standard deviation of gaps between logins

        # This creates 2 columns:
        # - active_days_per_week_pre_w3
        # - std_regularity_pre_w3

        def calculate_weekly_consistency(df_filtered, suffix, student_id_col='id_student', date_col='date', week_col='week'):
            weekly_days = df_filtered.groupby([student_id_col, week_col])[date_col].nunique()
            consistency = weekly_days.groupby(student_id_col).std()
            consistency = consistency.rename(f'active_days_per_week_{suffix}')
            return consistency

        def calculate_regularity_std_period(df_filtered, suffix, student_id_col='id_student', date_col='date'):
            regularity = df_filtered.groupby(student_id_col)[date_col].apply(
                lambda x: x.sort_values().diff().std()
            )
            regularity = regularity.rename(f'std_regularity_{suffix}')
            return regularity

        active_days_pre_w3 = calculate_weekly_consistency(df_pre_w3, 'pre_w3')
        std_regularity_pre_w3 = calculate_regularity_std_period(df_pre_w3, 'pre_w3')

        # Merge both metrics back to original dataframe
        df['active_days_per_week_pre_w3'] = df['id_student'].map(active_days_pre_w3)
        df['std_regularity_pre_w3'] = df['id_student'].map(std_regularity_pre_w3)

        student_level_columns.extend(['relative_submission_date','submission_type'])

    if 'diversity' in features:
        ## Add diversity of interaction features for pre-course through week 3
        # - VLE richness (number of different VLE types used) 
        # - Shannon entropy (overall diversity of interactions)

        # This creates 2 columns:
        # - vle_richness_pre_w3
        # - diversity_shannon_pre_w3

        from scipy.stats import entropy

        def shannon_entropy_calc(counts, vle_columns):
            counts = counts[vle_columns].values.astype(float)
            counts = counts[counts > 0]
            if len(counts) == 0:
                return 0
            proportions = counts / counts.sum()
            return entropy(proportions, base=2)

        def calculate_vle_metrics(df_filtered, suffix, vle_columns):
            student_vle = df_filtered.groupby('id_student')[vle_columns].sum() # Group by student and sum VLE activities for the period
            richness = (student_vle > 0).sum(axis=1).rename(f'vle_richness_{suffix}') # Calculate richness (number of VLE types used) 
            diversity = student_vle.apply(lambda row: shannon_entropy_calc(row, vle_columns), axis=1).rename(f'diversity_shannon_{suffix}')
            return richness, diversity

        vle_richness_pre_w3, diversity_shannon_pre_w3 = calculate_vle_metrics(df_pre_w3, 'pre_w3', vle_columns)

        # Merge both metrics back to original dataframe
        df['vle_richness_pre_w3'] = df['id_student'].map(vle_richness_pre_w3)
        df['diversity_shannon_pre_w3'] = df['id_student'].map(diversity_shannon_pre_w3)

        ## Drop students who already took course, and who withdrew before week 4
        df = df[df['num_of_prev_attempts'] == 0]
        df = df[~((df['final_result'] == 'Withdrawn') & (df['date_unregistration'] < n_weeks*7))]
        
        student_level_columns.extend(['vle_richness_pre_w3','diversity_shannon_pre_w3']) 

    if 'demographics' in features:
        student_level_columns.extend(['code_module','gender','region','highest_education','imd_band', 'age_band',
                             'num_of_prev_attempts','studied_credits', 'disability', 'final_result',
                             'date_registration','date_unregistration', 'module_presentation_length'])
    
    ## Collapse dataframe to one row per student
    # Check which columns exist in the dataframe
    available_columns = [col for col in student_level_columns if col in df.columns]

    # Group by student and presentation, taking the first value for each column
    # (student-level features should be the same across all rows for a given student)
    df_student_level = df.groupby(['id_student', 'code_presentation'])[available_columns].first().reset_index()

    ## save as csv
    if write_to:
        df_student_level.to_csv(write_to)

    return df_student_level