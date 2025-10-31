from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import random
import pandas as pd
import numpy as np

def data_preprocess(df):
    # helper function, prepares dataframe to be fit through pipeline.
    # takes as input: dataframe df
    # returns as output: prepared dataframe

    # Create binary target variable
    # 1 = Fail/Withdrawn 
    # 0 = Pass/Distinction 
    df['target'] = df['final_result'].isin(['Fail', 'Withdrawn']).astype(int)

    # Create a copy of the original df with the features we need
    # Not included: demographic features 
    # Not included: relative_submission_date (because it would cause student who never submitted to get dropped)
    feature_columns = [
        'code_module', 
        'content_focus_pre_w3',
        'collaborative_focus_pre_w3', 
        'active_days_per_week_pre_w3',
        'std_regularity_pre_w3', 
        'vle_richness_pre_w3',
        'diversity_shannon_pre_w3', 
        'submission_type',
        'total_vle_pre_w3'
    ]

    # Create working dataframe with features + target
    df_model = df[feature_columns + ['target']].copy()

    # Assign zeros to all interaction features that are null *except* submission_type
    columns_to_fill_zeros = [
        'active_days_per_week_pre_w3',
        'std_regularity_pre_w3',
    ]
    df_model[columns_to_fill_zeros] = df_model[columns_to_fill_zeros].fillna(0)
    return(df_model.dropna())

def model_traintest_split(df_clean, SEED=42):
    # helper function, prepares train / test split for dataframe
    # stratifies (at least initially) on both outcome and module-presentation
    # takes as input: dataframe df_clean, and (optionally) a random seed
    # returns as output: train / test sets 
    # note y_train is not returned; it is encoded as part of X_train.stratify

    # Separate features and target
    X = df_clean.drop('target', axis = 1)
    y = df_clean['target']

    # Create combined stratification column (target + code_module)
    stratify_column = y.astype(str) + '_' + X['code_module']

    ## Keep extra copy of column in X_train for further use in cross-validation splits
    X["stratify"] = stratify_column

    # Train/test split with stratification on target variable + module
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        stratify = stratify_column,
        random_state = SEED
    )

    X_test.drop(["stratify"], axis=1, inplace=True)

    return X_train, X_test, y_test

def model_initfit(df, models, SEED=42):
    # uses cross-validation to report metrics for models fitted on df

    # takes as input:
    # df: the data to which the model is to be fit
    # models: dictionary of pipelines to fit
    # SEED: random seed [for reproducibility]

    # returns as output: dictionary of metrics 
    # (specific metrics described below)

    random.seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_test = model_traintest_split(df)

    # Set X_train, y_train up for stratified splits during cross-validation
    y_train = X_train["stratify"]
    X_train.drop(["stratify"], axis=1, inplace=True)

    # Train and evaluate with cross-validation
    results = []
    skf = StratifiedKFold(n_splits=5)

    for model in models:
        auc, recall, precision, f1, overfit_gap = [], [], [], [], []
        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            X_tt = X_train.iloc[train_index]
            y_tt = y_train.iloc[train_index].apply(lambda x: int(x[0]))

            X_ho = X_train.iloc[test_index]
            y_ho = y_train.iloc[test_index].apply(lambda x: int(x[0]))   
            
            this_model = clone(models[model])
            this_model.fit(X_tt, y_tt)
        
            y_test_pred = this_model.predict(X_ho)
            y_train_pred = this_model.predict(X_tt)
        
            # Probabilities
            try:
                y_test_proba = this_model.predict_proba(X_ho)[:, 1]
                y_train_proba = this_model.predict_proba(X_tt)[:, 1]
                test_auc = roc_auc_score(y_ho, y_test_proba)
            except:
                test_auc = train_auc = np.nan

            # individual fold metrics
            auc += [test_auc]
            recall += [recall_score(y_ho, y_test_pred, zero_division=0)]
            precision += [precision_score(y_ho, y_test_pred)]
            f1 += [f1_score(y_ho, y_test_pred)]
            overfit_gap += [accuracy_score(y_tt, y_train_pred) - accuracy_score(y_ho, y_test_pred)]
    
        # Metrics averaged across all folds
        results.append({
            'Model': model,
            'Test ROC-AUC': np.mean(auc),
            'Test Recall': np.mean(recall),
            'Test Precision': np.mean(precision),
            'Test F1': np.mean(f1),
            'Overfitting Gap': np.mean(overfit_gap)
        })

    results_df = pd.DataFrame(results).sort_values('Test F1', ascending=False)
    return(results_df)

def model_tune(df, models, SEED=42):
    # uses GridSearchCV to do hyperparameter tuning for specified models

    # takes as input:
    # df: the data to which the model is to be fit
    # models: dictionary of pipelines to fit
    # SEED: random seed [for reproducibility]

    # returns as output: tuned models


    models_to_tune = ['Logistic Regression', 'Random Forest', 'Extra Trees']
    # Hyperparameter grids
    param_grids = {
        'Logistic Regression': {
            'classify__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            'classify__l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
        'Random Forest': {
            'classify__n_estimators': [50, 100, 150, 200, 300, 400, 500],
            'classify__max_depth': [4, 6, 8, 10],
            'classify__min_samples_split': [25, 50, 100],
            'classify__min_samples_leaf': [10, 25, 50]
        },
        'Extra Trees': {
            'classify__n_estimators': [50, 100, 150, 200, 300, 400, 500],
            'classify__max_depth': [4, 6, 8, 10],
            'classify__min_samples_split': [25, 50, 100],
            'classify__min_samples_leaf': [10, 25, 50]
        }
    }

    X_train, X_test, y_test = model_traintest_split(df)
    y_train = X_train["stratify"].apply(lambda x: int(x[0]))
    X_train.drop(["stratify"], axis=1, inplace=True)

    final_models = {}

    for model_name in models_to_tune:
        print(f"\nTuning {model_name}...")
        grid = GridSearchCV(clone(models[model_name]), param_grids[model_name], 
                            cv=5, scoring='recall', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        print(f"Best recall: {grid.best_score_:.4f}")
        final_models[model_name] = grid.best_estimator_

    return final_models