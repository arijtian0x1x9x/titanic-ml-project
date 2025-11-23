#!/usr/bin/env python3
"""
blend_predict.py

Train RF and GradientBoosting on full train.csv, blend probabilities,
and write submission_blend.csv using weights determined from CV:
    RF weight = 0.30
    GB weight = 0.70

Also saves the trained models to models/blend_models.joblib
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42
MODEL_DIR = "models"
OUT_SUBMISSION = "submission_blend.csv"
MODEL_SAVE = os.path.join(MODEL_DIR, "blend_models.joblib")

# --- conservative feature engineering (same as blend_search) ---
def extract_title(name):
    try:
        return name.split(",")[1].split(".")[0].strip()
    except:
        return "Unknown"

def prepare_df(df, train_medians=None):
    df = df.copy()
    # Fill Age with train median if provided, else median of df
    if train_medians and 'Age' in train_medians:
        df['Age'] = df['Age'].fillna(train_medians['Age'])
    else:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    # Title
    df['Title'] = df['Name'].apply(extract_title).replace(['Mlle','Ms'],'Miss').replace({'Mme':'Mrs'})
    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    # Deck coarse
    df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0] if isinstance(x, str) and len(x) else 'U')
    # Fill Embarked/fare if missing
    if train_medians and 'Fare' in train_medians:
        df['Fare'] = df['Fare'].fillna(train_medians['Fare'])
    else:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
    # AgeBin / FareBin optional (not used in this script)
    return df

def build_preprocessor(num_cols, cat_cols):
    # OneHotEncoder compatibility across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', ohe)
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')
    return pre

def main():
    # Load
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print("Loaded train", train.shape, "test", test.shape)

    # compute train medians to use for filling test
    train_medians = {
        'Age': train['Age'].median(),
        'Fare': train['Fare'].median()
    }

    # prepare dataframes
    train_p = prepare_df(train, train_medians)
    test_p = prepare_df(test, train_medians)

    # feature list (same as blend_search)
    features = [
        'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked',
        'Title','FamilySize','IsAlone','FarePerPerson','Deck'
    ]

    X = train_p[features].copy()
    y = train_p['Survived'].copy()
    X_test = test_p[features].copy()

    # preprocessor
    num_cols = ['Age','SibSp','Parch','Fare','FamilySize','FarePerPerson']
    cat_cols = ['Pclass','Sex','Embarked','Title','IsAlone','Deck']
    pre = build_preprocessor(num_cols, cat_cols)

    # model definitions (use tuned-ish hyperparams from previous runs)
    rf = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=400, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1))])
    gb = Pipeline([('pre', pre), ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE))])

    # Train on full data
    print("Training RandomForest on full training data...")
    rf.fit(X, y)
    print("Training GradientBoosting on full training data...")
    gb.fit(X, y)

    # Predict probabilities on test
    print("Predicting test probabilities...")
    proba_rf = rf.predict_proba(X_test)[:, 1]
    proba_gb = gb.predict_proba(X_test)[:, 1]

    # Blend weights discovered earlier
    w_rf = 0.30
    w_gb = 0.70
    blended_proba = w_rf * proba_rf + w_gb * proba_gb
    preds = (blended_proba >= 0.5).astype(int)

    # Build submission
    submission = pd.DataFrame({
        "PassengerId": test['PassengerId'],
        "Survived": preds
    })

    submission.to_csv(OUT_SUBMISSION, index=False)
    print(f"Saved submission to {OUT_SUBMISSION} (shape: {submission.shape})")

    # Save models and other artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        'rf_pipeline': rf,
        'gb_pipeline': gb,
        'blend_weights': {'rf': w_rf, 'gb': w_gb}
    }, MODEL_SAVE)
    print(f"Saved trained models to {MODEL_SAVE}")

if __name__ == "__main__":
    main()
