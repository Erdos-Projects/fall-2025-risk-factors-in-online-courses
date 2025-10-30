def student_selection(df,n_weeks,max_num_attempts): 

    # Dropping rows in which a student already took the course more than the defined max_num_attempts
    df = df[df['num_of_prev_attempts'] <= max_num_attempts]

    # Drop students who withdrew before week n+1 (i.e., withdrew during weeks 1 to n or pre-course)
    df = df[~((df['final_result'] == 'Withdrawn') & (df['date_unregistration'] < (n_weeks*7)))]

    return df


def course_selection(df, assessments, n_weeks): 

    # Rename 'date' column in assessments to 'due_date'
    assessments = assessments.rename(columns={'date': 'due_date'})

    # Determine what is the first assessment per module, presentation
    first_assessment = (
        assessments
        .sort_values('due_date', ascending=True)  # sort so earliest comes first
        .groupby(['code_module', 'code_presentation'], as_index=False)
        .first()  # keep the first row per group (i.e., earliest due_date)
    )

    # Determine which module-presentations have a first assessment in the first 3 weeks (<21 days)
    idx = (
        first_assessment
        .groupby(['code_module', 'code_presentation'])['due_date']
        .max()                     # find the max due_date per module-presentation
        .loc[lambda x: x < (n_weeks*7)]     # keep only those < 7*n_weeks days
        .index
    )

    presentations_to_keep = first_assessment.set_index(['code_module', 'code_presentation']).loc[idx].reset_index()
    selected_courses = presentations_to_keep[['code_module','code_presentation']].drop_duplicates()

    print("Selected courses:")
    print(selected_courses)

    df = df.merge(
        selected_courses,
        on=['code_module', 'code_presentation'],
        how='inner'
    )

    return df 

def check_samplesize(df, df_subset):
    import pandas as pd

    # Check how many students were excluded 
    # Count total students per module-presentation
    total_students_per_modpres = (
        df.groupby(['code_module', 'code_presentation'])['id_student']
        .nunique()
        .reset_index(name='total_students')
    )

    # Count only students retained after filtering
    subset_students_per_modpres = (
        df_subset.groupby(['code_module', 'code_presentation'])['id_student']
        .nunique()
        .reset_index(name='subset_students')
    ).round(0)

    # Merge and compute percentage
    excluded_summary = total_students_per_modpres.merge(
        subset_students_per_modpres,
        on=['code_module', 'code_presentation'],
        how='left'
    )

    # Fill missing counts with 0 and compute percentage
    excluded_summary['pct_retained'] = (
        excluded_summary['subset_students'] / excluded_summary['total_students'] * 100
    ).round(1)

    # Show result
    pd.set_option('display.width', 2000)
    print(excluded_summary.to_string(index=False))
