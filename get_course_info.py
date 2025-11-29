def course_meta_info(df, meta=['length','num_students','num_assessments','percent_outcomes'], output=True):
    # inputs: 
    # df = dataframe
    # meta = list of metadata info to calculate
    # - length: length of module-presentation in days
    # - num_students: number of students in each module-presentation
    # - num_assessments: number of assessments in each module-presentation
    # - percent_outcomes: percentage of students within each final_result in each module-presentation 
    # output = whether to print table, default is True 
    # outputs: 
    # printed table if output = True
    # returns table 

    df_course_meta = df[["code_module", "code_presentation"]].drop_duplicates().reset_index()

    if 'length' in meta:     
        course_lengths = (
            df
            .groupby(["code_module", "code_presentation"], as_index=False, observed = False)
            ["module_presentation_length"]
            .first()
        )
        df_course_meta = df_course_meta.merge(course_lengths, on=["code_module", "code_presentation"], how="left")

    if 'num_students' in meta:
        students_in_course = (
            df
            .groupby(["code_module", "code_presentation"], as_index=False, observed = False)
            ["id_student"]
            .nunique()
            .rename(columns={"id_student": "num_students"})
        )
        df_course_meta = df_course_meta.merge(students_in_course, on=["code_module", "code_presentation"], how="left")

    if 'num_assessments' in meta: 
        assessments_in_course = (
            df
            .groupby(["code_module", "code_presentation","assessment_type"], observed = False)
            ["id_assessment"]
            .nunique()
            .reset_index(name="num_unique_assessments")
        )

        assessments_in_course_wide = (
            assessments_in_course
            .pivot_table(
                index=["code_module", "code_presentation"],
                columns="assessment_type",
                values="num_unique_assessments",
                fill_value=0,  # fills missing combinations with 0
                observed = False)
            .reset_index()
        )

        assessments_in_course_wide.columns.name = None  # remove the pivot column name
        
        df_course_meta = df_course_meta.merge(assessments_in_course_wide, on=["code_module", "code_presentation"], how="left")

    if 'percent_outcomes' in meta: 
        outcomes_in_course = (
            df
            .groupby(["code_module", "code_presentation","final_result"], observed = False)
            ["id_student"]
            .nunique()
            .reset_index(name="num_students")
        )

        outcomes_in_course_wide = (
            outcomes_in_course
            .pivot_table(
                index=["code_module", "code_presentation"],
                columns="final_result",
                values="num_students",
                fill_value=0,  # fills missing combinations with 0
                observed = False)
            .reset_index()
        )

        outcomes_in_course_wide.columns.name = None  # remove the pivot column name

        # cols = [c for c in ['Distinction', 'Pass', 'Withdrawn', 'Fail'] if c in outcomes_in_course_wide.columns]

        outcomes_in_course_wide[['Distinction','Pass','Withdrawn','Fail']] = (
            outcomes_in_course_wide[['Distinction','Pass','Withdrawn','Fail']]
            .div(outcomes_in_course_wide[['Distinction','Pass','Withdrawn','Fail']].sum(axis=1), axis=0)
            * 100
        ).round(1)

        outcomes_in_course_wide.rename(
            columns={col: f"{col}%" for col in ['Distinction','Pass','Withdrawn','Fail']},
            inplace=True
        )

        df_course_meta = df_course_meta.merge(outcomes_in_course_wide, on=["code_module", "code_presentation"], how="left").drop(columns="index")

    if output: 
        print(df_course_meta)

    return(df_course_meta)