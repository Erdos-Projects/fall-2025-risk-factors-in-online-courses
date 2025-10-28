def assessment_features(df,assessments,features=['submission'],n_weeks=3,write_to=None):
    # parameters: 
    # df is the merged dataframe containing all necessary variables
    # features is a list of feature types we wish to calculate. all are included by default
    # n_weeks determines the time period to be selected from the beginning of the data set
    # write_to can be used to specify a file path if the user wishes to save the final dataframe in a csv

    import numpy as np

    # Rename 'date' column in assessments to 'due_date'
    assessments = assessments.rename(columns={'date': 'due_date'})

    features = [item.lower() for item in features] # avoid case-sensitive errors
  
    assessment_columns = ['id_assessment','is_banked','score','assessment_type','due_date','weight']
    course_student_assessment_cols = ['code_module','code_presentation','module_presentation_length', 'date', 
                                'id_student','final_result','date_registration','date_unregistration'
                                ] + assessment_columns   
    
    # Create a "week" column relative to each code_presentation's starting date.
    # Dates 0.0 - 6.0 are week 1, 7.0 - 13.0 are week 2, etc. 
    # The weeks *before* date 0.0 are assigned to negative values.
    df['week'] = df.groupby('code_presentation')['date'].transform( # create "week" column
        lambda x: ((x // 7) + 1).where(x >= 0, x // 7))
    
    # Keep columns relevant to assessments and drop rows with no assessment 
    df_assessment = df[course_student_assessment_cols].dropna(subset=["id_assessment"])

    # Determine what is the first assessment per module, presentation
    first_assessment = (
        assessments
        .sort_values('due_date', ascending=True)  # sort so earliest comes first
        .groupby(['code_module', 'code_presentation'], as_index=False)
        .first()  # keep the first row per group (i.e., earliest due_date)
    )

    # Only keep first assessment rows for each student
    df_first_assessment = df_assessment.merge(
        first_assessment[['code_module', 'code_presentation', 'id_assessment']],
        on=['code_module', 'code_presentation', 'id_assessment'],
        how='inner'
    )
    df_first_assessment = df_first_assessment.rename(columns={'due_date': 'first_due_date'}) # rename column

    # Add students who did not turn in a first assessment back into df_first_assessment and following assessment-related columns will be NaN:
    # - date	
    # - is_banked	
    # - score	
    # - week

    # Step 1: Start with all unique students from df_vle (keep all student info columns)
    students_all = df[[
        'id_student', 'code_module', 'code_presentation',
        'final_result', 'date_registration', 'date_unregistration',
        'module_presentation_length'
    ]].drop_duplicates()

    # Step 2: Merge with df_first_assessment (left join keeps all VLE students)
    df_first_assessment_full = students_all.merge(
        df_first_assessment,
        on=['id_student', 'code_module', 'code_presentation',
        'final_result', 'date_registration', 'date_unregistration',
        'module_presentation_length'],
        how='left',
        suffixes=('', '_first')
    )

    # Step 3: Add binary column to indiciate who submitted first assessment (1=yes, 0=no)
    df_first_assessment_full['submitted_first_assessment'] = (
        df_first_assessment_full['id_assessment'].notna().astype(int)
    )

    # Step 4: Merge in module-level first assessment info
    df_first_assessment_full = df_first_assessment_full.merge(
        first_assessment[['code_module', 'code_presentation', 'id_assessment','assessment_type', 'due_date', 'weight']],
        on=['code_module', 'code_presentation'],
        how='left',
        suffixes=('', '_expected')
    )

    # Step 5: Use expected info for missing students
    df_first_assessment_full['id_assessment'] = df_first_assessment_full['id_assessment'].fillna(
        df_first_assessment_full['id_assessment_expected']
    )
    df_first_assessment_full['assessment_type'] = df_first_assessment_full['assessment_type'].fillna(
        df_first_assessment_full['assessment_type_expected']
    )
    df_first_assessment_full['first_due_date'] = df_first_assessment_full['first_due_date'].fillna(
        df_first_assessment_full['due_date']
    )
    df_first_assessment_full['weight'] = df_first_assessment_full['weight'].fillna(
        df_first_assessment_full['weight_expected']
    )

    # Clean up helper columns
    df_first_assessment_full.drop(columns=['id_assessment_expected','assessment_type_expected', 'due_date', 'weight_expected'], inplace=True)

    # Add submission related engineered features for FIRST ASSESSMENT only:
    if 'submission_' in features:
        df_first_assessment_full['relative_submission_date'] = df_first_assessment_full['first_due_date'] - df_first_assessment_full['date']
        df_first_assessment_full['submission_type'] = np.select(
            [
                df_first_assessment_full['relative_submission_date'].isna(),
                df_first_assessment_full['relative_submission_date'] <= 0, # early & on-time
                # df_first_assessment_full['relative_submission_date'] == 0,
                df_first_assessment_full['relative_submission_date'] > 0
            ],
            [
                'Never',
                'Early',
                # 'On-time',
                'Late'
            ],
            default='Unknown'
        )

    return df_first_assessment_full

def vle_features(df,features=['total','focus','regularity','diversity','demographics'],
                       n_weeks=3,write_to=None):
    # parameters: 
    # df is the merged dataframe containing all necessary variables
    # features is a list of feature types we wish to calculate. all are included by default
    # n_weeks determines the time period to be selected from the beginning of the data set
    # write_to can be used to specify a file path if the user wishes to save the final dataframe in a csv
    
    import numpy as np

    features = [item.lower() for item in features] # avoid case-sensitive errors

    # course_student_vle_cols = df.columns[:24].tolist() + df.columns[30:].tolist() 
    vle_columns = ['quiz', 'questionnaire', 'externalquiz', 'oucontent', 'page', 
                   'resource', 'url', 'homepage', 'glossary', 'subpage', 'folder', 
                   'forumng', 'oucollaborate', 'ouelluminate', 'ouwiki', 'sharedsubpage', 
                   'dataplus','repeatactivity', 'dualpane', 'htmlactivity']
    course_student_vle_cols = ['code_module','code_presentation','module_presentation_length', 'date', 
                                'id_student','final_result','date_registration','date_unregistration'
                                ] + vle_columns
    
    # Keep columns relevant to vle interactions and drop duplicate rows that arose if students submitted multiple assessments on a given date
    df_vle = df[course_student_vle_cols].drop_duplicates()
    # Keep rows where at least one VLE column is nonzero and not NaN
    df_vle = df_vle[df_vle[vle_columns].fillna(0).sum(axis=1) > 0] # takes care of instances where there is NaN in a column (I noticed it in quiz) and everything else is 0

    # Create a "week" column relative to each code_presentation's starting date.
    # Dates 0.0 - 6.0 are week 1, 7.0 - 13.0 are week 2, etc. 
    # The weeks *before* date 0.0 are assigned to negative values.
    df_vle['week'] = df_vle.groupby('code_presentation')['date'].transform( # create "week" column
        lambda x: ((x // 7) + 1).where(x >= 0, x // 7))
    
    df_vle_early = df_vle[df_vle['week'] <= n_weeks] # Filter to pre-course through week n and sum interactions per student

    # Add students who did not have vle interactions during first 3 weeks back into df_vle_early and all vle columns will be 0:
    # - date is assigned NaN	
    # - week is assigned NaN

    # Step 1: Start with all unique students from df_vle (keep all student info columns)
    students_all = df[[
        'id_student', 'code_module', 'code_presentation',
        'final_result', 'date_registration', 'date_unregistration',
        'module_presentation_length'
    ]].drop_duplicates()

    # Step 2: Merge with df_vle_early (left join keeps all VLE students)
    df_vle_early_full = students_all.merge(
        df_vle_early,
        on=['id_student', 'code_module', 'code_presentation',
        'final_result', 'date_registration', 'date_unregistration',
        'module_presentation_length'],
        how='left',
        suffixes=('', '_first')
    )

    # Step 3: Assign 0 to all vle interactions for those students 
    df_vle_early_full.loc[df_vle_early_full['date'].isna(), vle_columns] = 0

    # VLE feature calculations #################################################
    student_level_columns = [] # empty list of student-level characteristics to be appended

    if 'total' in features:
        # Add feature that sums total VLE interactions per student through week 3
        # - total_vle_pre_w3

        student_totals = df_vle_early_full.groupby([
            'id_student','code_module','code_presentation'
            ])[vle_columns].sum().reset_index()
        student_totals['total_vle_pre_w3'] = student_totals[vle_columns].sum(axis=1)

        # Merge these features back into dataframe
        df_vle_early_full = df_vle_early_full.merge(
            student_totals[['id_student', 'code_module','code_presentation','total_vle_pre_w3']],
            on=['id_student', 'code_module','code_presentation'],
            how='left')
        
        student_level_columns.extend(['total_vle_pre_w3'])

    if 'focus' in features:
        # Add VLE interaction category features:
        # - Content type interaction (percentage)
        # - Collaborative type interaction (percentage)

        # This creates 2 columns for pre-course through week 3:
        # - collaborative_focus_pre_w3
        # - content_focus_pre_w3

        content_types = ['oucontent', 'page', 'resource', 'url', 'homepage', 'glossary', 'subpage', 'folder']
        collaborative_types = ['forumng', 'oucollaborate', 'ouelluminate', 'ouwiki', 'sharedsubpage']

        student_totals = df_vle_early_full.groupby([
            'id_student','code_module','code_presentation'
            ])[vle_columns].sum().reset_index()

        student_totals['content_focus_pre_w3'] = student_totals[content_types].sum(axis=1) / student_totals[vle_columns].sum(axis=1)
        student_totals['content_focus_pre_w3'] = student_totals['content_focus_pre_w3'].fillna(0)

        student_totals['collaborative_focus_pre_w3'] = student_totals[collaborative_types].sum(axis=1) / student_totals[vle_columns].sum(axis=1)
        student_totals['collaborative_focus_pre_w3'] = student_totals['collaborative_focus_pre_w3'].fillna(0)

        # Merge these features back into dataframe
        df_vle_early_full = df_vle_early_full.merge(
            student_totals[['id_student', 'code_module','code_presentation','content_focus_pre_w3','collaborative_focus_pre_w3']],
            on=['id_student', 'code_module','code_presentation'],
            how='left')        
        
        student_level_columns.extend(['content_focus_pre_w3','collaborative_focus_pre_w3'])        

    if 'regularity' in features:

        # Add regularity features for pre-course through week 3:
        # - Standard deviation of active days per week
        # - Standard deviation of gaps between days with interactions

        # This creates 2 columns:
        # - active_days_per_week_pre_w3
        # - std_regularity_pre_w3

        def calculate_weekly_consistency(df_filtered, suffix, student_id_col='id_student', date_col='date', week_col='week'):
            weekly_days = df_filtered.groupby([student_id_col, week_col])[date_col].nunique()
            consistency = weekly_days.groupby(student_id_col).std()
            consistency = consistency.rename(f'active_days_per_week_pre_w3_{suffix}')
            return consistency

        def calculate_regularity_std_period(df_filtered, suffix, student_id_col='id_student', date_col='date'):
            regularity = df_filtered.groupby(student_id_col)[date_col].apply(
                lambda x: x.sort_values().diff().std())
            regularity = regularity.rename(f'std_regularity_{suffix}')
            return regularity

        active_days_pre_w3 = calculate_weekly_consistency(df_vle_early_full, 'pre_w3')
        active_days_pre_w3

        std_regularity_pre_w3 = calculate_regularity_std_period(df_vle_early_full, 'pre_w3')
        std_regularity_pre_w3 

        # Merge both metrics back to original dataframe
        df_vle_early_full['active_days_per_week_pre_w3'] = df_vle_early_full['id_student'].map(active_days_pre_w3)
        df_vle_early_full['std_regularity_pre_w3'] = df_vle_early_full['id_student'].map(std_regularity_pre_w3)

        student_level_columns.extend(['active_days_per_week_pre_w3','std_regularity_pre_w3'])

    if 'diversity' in features:

        # Add diversity of interaction features for pre-course through week 3
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
            student_totals = df_filtered.groupby([
                'id_student','code_module','code_presentation'
                ])[vle_columns].sum().reset_index() # Group by student and sum VLE activities for the period
            richness = (student_totals[vle_columns] > 0).sum(axis=1).rename(f'vle_richness_{suffix}') # Calculate richness (number of VLE types used) 
            diversity = student_totals[['id_student'] + vle_columns].apply(
                lambda row: shannon_entropy_calc(row, vle_columns), axis=1).rename(f'diversity_shannon_{suffix}')
            return richness, diversity
        
        vle_richness_pre_w3, diversity_shannon_pre_w3 = calculate_vle_metrics(df_vle_early_full, 'pre_w3', vle_columns)       

        # Merge both metrics back to original dataframe
        df_vle_early_full['vle_richness_pre_w3'] = df_vle_early_full['id_student'].map(vle_richness_pre_w3)
        df_vle_early_full['diversity_shannon_pre_w3'] = df_vle_early_full['id_student'].map(diversity_shannon_pre_w3)

        student_level_columns.extend(['vle_richness_pre_w3','diversity_shannon_pre_w3']) 
    

    # Collapse dataframe to one row per student
    # Check which columns exist in the dataframe
    student_level_columns.extend(['final_result','date_registration','date_unregistration', 'module_presentation_length'])
    available_columns = [col for col in student_level_columns if col in df_vle_early_full.columns]

    # Group by student-module-presentation, taking the first value for each column
    # (since student-level features should be the same across all rows for a given student)
    df_vle_early_student_level = df_vle_early_full.groupby(['id_student','code_module','code_presentation'])[available_columns].first().reset_index()
    # df_vle_pre_w3_student_level = df_vle_pre_w3_student_level.drop(columns=['date', 'week'] + vle_columns)

    # ## save as csv
    # if write_to:
    #     df_student_level.to_csv(write_to)

    return df_vle_early_student_level