import pandas as pd, numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import pyarrow.dataset as ds
import glob
import zipfile

def unzip_data(filepath):
    # inputs: file path of zipped file
    # outputs: dataframe for each csv in the zip file

    zipped = zipfile.ZipFile(filepath) 

    assessments = pd.read_csv(zipped.open('assessments.csv')) # info about the assessments
    courses = pd.read_csv(zipped.open('courses.csv')) # info on all available modules and their presentations
    studentAssessments = pd.read_csv(zipped.open('studentAssessment.csv')) # contains results of students' assessments 
    studentInfo = pd.read_csv(zipped.open('studentInfo.csv')) # contains emographic information about the students together with their results
    studentRegistration = pd.read_csv(zipped.open('studentRegistration.csv')) # info about the time when the student registered (and unregistered, if applicable)
    studentVLE = pd.read_csv(zipped.open('studentVle.csv')) # contains information about each student's interactions with the materials in the VLE
    VLEdata = pd.read_csv(zipped.open('vle.csv')) # info about the available materials in the VLE

    return(assessments, courses, studentAssessments, studentInfo, studentRegistration, studentVLE, VLEdata)

def merge_data(filepath, write_to=None):
    # inputs: 
    # filepath = file path of zipped file
    # write_to  = file path to write csv file containing merged data, default is None (current directory)
    # outputs: 
    # dataframe 
    # saved csv

    zipped = zipfile.ZipFile(filepath) 

    assessments = pd.read_csv(zipped.open('assessments.csv')) # info about the assessments
    courses = pd.read_csv(zipped.open('courses.csv')) # info on all available modules and their presentations
    studentAssessments = pd.read_csv(zipped.open('studentAssessment.csv')) # contains results of students' assessments 
    studentInfo = pd.read_csv(zipped.open('studentInfo.csv')) # contains emographic information about the students together with their results
    studentRegistration = pd.read_csv(zipped.open('studentRegistration.csv')) # info about the time when the student registered (and unregistered, if applicable)
    studentVLE = pd.read_csv(zipped.open('studentVle.csv')) # contains information about each student's interactions with the materials in the VLE
    VLEdata = pd.read_csv(zipped.open('vle.csv')) # info about the available materials in the VLE
   
    # Both assessments and studentVLE have 'date' column but are not the same variable. 
    # Rename 'date' column in assessments to 'due_date'
    assessments = assessments.rename(columns={'date': 'due_date'})

    # Merge studentInfo & studentRegistration dataframes & courses --> df_studentInfo
    common_columns = list(set(studentInfo.columns) & set(studentRegistration.columns)) # common_columns = ['code_module','code_presentation','id_student']
    df_studentInfo=pd.merge(studentInfo, studentRegistration, how='outer', on=common_columns)
    common_columns = list(set(courses.columns) & set(df_studentInfo.columns)) # common_columns = ['code_module', 'code_presentation']
    df_studentInfo=pd.merge(courses, df_studentInfo, how='outer', on=common_columns) # adds column module_presentation_length to df_studentInfo

    # Merge assessments & studentAssessment dataframes --> df_assessment
    common_columns = list(set(assessments.columns) & set(studentAssessments.columns)) # common_columns = ['id_assessment']
    df_assessment=pd.merge(studentAssessments, assessments, how='left', on=common_columns)

    # Merge studentVLE & VLEdata dataframes --> df_vle
    common_columns = list(set(studentVLE.columns) & set(VLEdata.columns)) # common_columns = ['id_site', 'code_presentation', 'code_module']
    df_vle = pd.merge(studentVLE,VLEdata, how='left', on=common_columns)

    # Collapse id_site in df_vle such that all interactions for each activity_type occurring on the same date are summed. 
    # This is a memory-intensive operation, so a workaround using parquet files is provided below #################################
    
    # Create parquet files for each (code_module, code_presentation) combination
    output_path = "Data/parquet_files/" #temp_destination # "data/OULAD/studentVLE_partitioned/"

    for module, pres in df_vle[["code_module", "code_presentation"]].drop_duplicates().itertuples(index=False):
    
        df_part = df_vle[(df_vle["code_module"] == module) & (df_vle["code_presentation"] == pres)]
    
        folder = os.path.join(output_path, 
                          f"code_module={module}",
                          f"code_presentation={pres}")
        os.makedirs(folder, exist_ok=True)

        # Enforce consistent dtypes
        df_part = df_part.astype({
            "id_student": "int32",
            "date": "int16",
            "code_module": "string",
            "code_presentation": "string"
        })
    
        file_path = os.path.join(folder, "data.parquet")
        df_part.to_parquet(file_path, index=False, engine="pyarrow", use_dictionary=False)

    print("✅ Data partitioned by code_module & code_presentation and written into parquet files stored in Data/parquet_files")


    # Read in each (code_module, code_presentation) parquet file to perform groupby and pivot operations 
    # (operations are too memory intensive to do on the entire dataset at once)
    # output is saved as new parquet file

    dataset = ds.dataset("Data/parquet_files/", format="parquet",partitioning="hive") #dataset = ds.dataset("data/OULAD/studentVLE_partitioned/", format="parquet",partitioning="hive")

    modules = dataset.partitioning.dictionaries[0].to_pylist()  # code_module values
    presentations = dataset.partitioning.dictionaries[1].to_pylist()  # code_presentation values

    input_base = "Data/parquet_files/" # "data/OULAD/studentVLE_partitioned"
    output_base = "Data/aggregated/" # "data/OULAD/studentVLE_agg_partitioned"

    # Find all parquet files in the partitioned structure
    paths = glob.glob(f"{input_base}/code_module=*/code_presentation=*/data.parquet")
    paths = [os.path.normpath(p) for p in paths]

    for path in paths:
        # Parse partition values from folder names
        parts = path.split(os.sep)
        code_module = parts[-3].split("=")[1]
        code_presentation = parts[-2].split("=")[1]

        # Read file
        df = pd.read_parquet(path)

        # Group and sum
        agged = (
            df.groupby(
                ["id_student", "code_module", "code_presentation", "date", "activity_type"],
                as_index=False, 
                observed=True
            )["sum_click"]
            .sum()
        )
        # Pivot so activity_type becomes columns
        pivoted = agged.pivot_table(
            index=["id_student", "code_module", "code_presentation", "date"],
            columns="activity_type",
            values="sum_click",
            fill_value=0, 
            observed = True
        ).reset_index()

        # Ensure activity_type columns are part of flat column index
        pivoted.columns.name = None

        # Reindex to ensure *all* activity types are present, even if missing
        all_activities = df_vle["activity_type"].unique().tolist()
        pivoted = pivoted.reindex(
            columns=["id_student", "code_module", "code_presentation", "date", *all_activities],
            fill_value=0
        )

        # Enforce consistent dtypes
        pivoted = pivoted.astype({
            "id_student": "int32",
            "date": "int16",
            "code_module": "string",
            "code_presentation": "string"
        })

        # Force activity columns to int16
        for col in pivoted.columns:
            if col not in ["id_student", "date", "code_module", "code_presentation"]:
                pivoted[col] = pivoted[col].astype("int16")    

        # Write result to same Hive-style folder structure
        out_dir = os.path.join(
            output_base,
            f"code_module={code_module}",
            f"code_presentation={code_presentation}"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data.parquet")
        pivoted.to_parquet(out_path, engine="pyarrow", index=False)

    print("✅ Aggregated data partitioned by code_module & code_presentation and written into parquet files stored in Data/aggregated")

    # Read in all parquet files to recreate full VLE interactions data table that is now grouped and pivoted
    dataset = ds.dataset(output_base, format="parquet")
    df_vle_agg = dataset.to_table().to_pandas()

    #########################################################################################

    # Make sure data types are consistent and downcast where possible
    df_vle_agg = df_vle_agg.astype({
        "code_module": "category", 
        "code_presentation": "category"
    })

    pd.to_numeric(df_studentInfo.date_registration, errors='coerce')
    df_studentInfo = df_studentInfo.astype({
        "id_student": "int32",
        "module_presentation_length": "int32",
        "num_of_prev_attempts": "int32",
        "studied_credits": "int32",
        "date_registration": "float32",     # safer with NaNs
        "date_unregistration": "float32"
    })

    categorical_cols = [
        "code_module", "code_presentation", 
        "gender", "region", "highest_education", "imd_band",
        "age_band", "disability", "final_result"
    ]
    df_studentInfo[categorical_cols] = df_studentInfo[categorical_cols].astype("category")

    df_assessment = df_assessment.astype({
        "id_assessment": "int32",
        "id_student": "int32",
        "date_submitted": "int32",
        "is_banked": "int8",
        "score": "float32",
        "due_date": "float32",              
        "weight": "float32",
        "code_module": "category", 
        "code_presentation": "category",
        "assessment_type": "category"
    })
    
    # Merge df_vle_agg & df_assessment --> df_vle_assessment (match assessment date_submitted with VLE date) 
    df_vle_assessment = pd.merge(df_vle_agg, df_assessment,
        left_on=["id_student", "code_module", "code_presentation", "date"],
        right_on=["id_student", "code_module", "code_presentation", "date_submitted"],
        how="outer"  # keeps all df_vle_agg rows
    )
    # Fill NaNs in VLE activity columns with 0
    vle_cols = ["forumng", "homepage", "oucontent", "subpage", "url", "resource",
                "dataplus", "glossary", "oucollaborate", "ouelluminate", 
                "sharedsubpage", "questionnaire", "page", "externalquiz", 
                "ouwiki", "dualpane", "repeatactivity", "folder", "htmlactivity"]

    for col in vle_cols:
        if col in df_vle_assessment.columns:
            df_vle_assessment[col] = df_vle_assessment[col].fillna(0).astype("int16")

    # Combine date columns into a single one
    df_vle_assessment["date"] = df_vle_assessment["date"].combine_first(df_vle_assessment["date_submitted"])
    df_vle_assessment = df_vle_assessment.drop(columns=["date_submitted"])
    df_vle_assessment.reset_index(drop=True, inplace=True)

    # Merge df_vle_assessment & df_studentInfo --> df 
    common_columns = list(set(df_vle_assessment.columns) & set(df_studentInfo.columns)) # common_columns = ['code_module', 'id_student', 'code_presentation']
    df=pd.merge(df_vle_assessment, df_studentInfo, how='left', on=common_columns)

    if write_to:
        df.to_csv(write_to, index = False)

    print("✅ Merged data stored in Data/merged_data.csv")

    return(df)
