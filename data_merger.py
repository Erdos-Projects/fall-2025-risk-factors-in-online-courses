def data_merger(filepath,write_to=None):
    import pandas as pd
    import zipfile
    zipped = zipfile.ZipFile(filepath) 

    assessments = pd.read_csv(zipped.open('assessments.csv')) # info about the assessments
    courses = pd.read_csv(zipped.open('courses.csv'))
    studentAssessments = pd.read_csv(zipped.open('studentAssessment.csv'))
    studentInfo = pd.read_csv(zipped.open('studentInfo.csv'))
    studentRegistration = pd.read_csv(zipped.open('studentRegistration.csv'))
    studentVLE = pd.read_csv(zipped.open('studentVle.csv')) # virtual learning environment
    VLEdata = pd.read_csv(zipped.open('vle.csv'))

    assessments = assessments.rename(columns={'date': 'due_date'})

    # Merge studentInfo & studentRegistration dataframes 
    common_columns = list(set(studentInfo.columns) & set(studentRegistration.columns)) # ['code_module','code_presentation','id_student']
    df_studentInfo=pd.merge(studentInfo, studentRegistration, how='outer', on=common_columns)

    # checks on merge
    assert studentInfo.shape[1] + studentRegistration.shape[1] - len(common_columns) == df_studentInfo.shape[1]
    assert studentInfo.shape[0] == studentRegistration.shape[0] == df_studentInfo.shape[0]

    # Merge courses with above (adds column module_presentation_length)
    common_columns = list(set(courses.columns) & set(df_studentInfo.columns)) # ['code_module', 'code_presentation']
    df_studentInfo=pd.merge(courses, df_studentInfo, how='outer', on=common_columns)

    # Merge assessments, studentAssessment dataframes 
    common_columns = list(set(assessments.columns) & set(studentAssessments.columns)) # ['id_assessment']
    df_assessment=pd.merge(studentAssessments, assessments, how='left', on=common_columns)

    # Merge studentVLE, VLEdata dataframes 
    common_columns = list(set(studentVLE.columns) & set(VLEdata.columns)) # ['id_site', 'code_presentation', 'code_module']
    df_vle = pd.merge(studentVLE,VLEdata, how='left', on=common_columns)

    # Merge df_studentInfo, df_assessment dataframes 
    common_columns = list(set(df_studentInfo.columns) & set(df_assessment.columns)) # ['code_module', 'id_student', 'code_presentation']
    df_student_assessment=pd.merge(df_assessment, df_studentInfo, how='left', on=common_columns)

    # Downcast to smaller data types based on your column stats
    df_vle = df_vle.astype({
        "id_student": "int32",
        "id_site": "int32",
        "date": "int16",
        "sum_click": "int16",
        "week_from": "Int8", # nullable integer type (Int8, Int16, Int32) instead of plain numpy ints allows for NaNs
        "week_to": "Int8" # nullable integer type (Int8, Int16, Int32) instead of plain numpy ints allows for NaNs
    })

    df_vle["code_module"] = df_vle["code_module"].astype("category")
    df_vle["code_presentation"] = df_vle["code_presentation"].astype("category")
    df_vle["activity_type"] = df_vle["activity_type"].astype("category")

    df_student_assessment = df_student_assessment.astype({
        "id_assessment": "int32",
        "id_student": "int32",
        "date_submitted": "int16",
        "is_banked": "int8",
        "module_presentation_length": "int16",
        "num_of_prev_attempts": "int8",
        "studied_credits": "int16",
        "score": "float32",
        "due_date": "Int16",              # could use 'Int16' if you want integer + NaNs
        "weight": "float32",
        "date_registration": "float32",     # safer with NaNs
        "date_unregistration": "float32"
    })

    categorical_cols = ["code_module", "code_presentation", "assessment_type","gender", "region", 
                        "highest_education", "imd_band","age_band", "disability", "final_result"]
    df_student_assessment[categorical_cols] = df_student_assessment[categorical_cols].astype("category")

    # Merge df_student_assessment, df_vle dataframes 
    common_columns = list(set(df_student_assessment.columns) & set(df_vle.columns)) # ['code_module', 'id_student', 'code_presentation']
    df=pd.merge(df_vle, df_student_assessment, how='left', on=common_columns)  # memory issue??

    if write_to:
        df.to_csv(write_to)

    return(df)
