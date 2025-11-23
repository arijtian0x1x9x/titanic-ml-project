#!/usr/bin/env python3
"""
titanic_model_enhanced.py

Enhanced Titanic pipeline:
- Model-based Age imputation (RandomForestRegressor)
- Stronger feature engineering (Titles, Deck, Family, bins)
- Logistic Regression tuning (regularization)
- RandomForest randomized search tuning
- XGBoost and LightGBM models included if installed
- StackingClassifier to blend models
- Produces submission.csv and saves final pipeline

Usage:
    python titanic_model_enhanced.py        # default (may run tuning)
    python titanic_model_enhanced.py --fast # skip heavy hyperparameter searches
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone
import joblib
import time
import json

# optional boosters
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

RANDOM_STATE = 42
MODEL_DIR = "models"
SUBMISSION_FN = "submission.csv"
FINAL_MODEL_FN = os.path.join(MODEL_DIR, "final_stacked_model.joblib")


# -------------------------
# Helpers & feature funcs
# -------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Required file not found: {path}")
        sys.exit(1)
    return pd.read_csv(path)


def extract_title(name: str):
    try:
        title = name.split(",")[1].split(".")[0].strip()
        return title
    except Exception:
        return "Unknown"


def standardize_title(title: str):
    t = title
    if t in ['Mlle', 'Ms']:
        return 'Miss'
    if t == 'Mme':
        return 'Mrs'
    if t in ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']:
        return 'Rare'
    return t


def feature_engineer_base(df: pd.DataFrame) -> pd.DataFrame:
    """Basic engineered features independent of age imputation"""
    df = df.copy()

    # Title
    df['Title'] = df['Name'].apply(extract_title).apply(standardize_title)

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Deck from Cabin
    df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0] if (isinstance(x, str) and len(x) > 0) else 'U')

    # Fill Embarked with most common
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

    return df


def impute_age_model(train_df: pd.DataFrame, test_df: pd.DataFrame, features_for_age=None):
    """
    Impute Age using a RandomForestRegressor trained on rows with Age present.
    Returns new train and test DataFrames with Age imputed.
    """
    train = train_df.copy()
    test = test_df.copy()

    if features_for_age is None:
        features_for_age = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Title', 'Deck', 'Embarked']

    # Prepare helper to convert categorical to numeric for the regressor
    def prepare_for_regression(df):
        X = pd.DataFrame()
        X['Pclass'] = df['Pclass'].astype(float)
        X['SibSp'] = df['SibSp'].astype(float)
        X['Parch'] = df['Parch'].astype(float)
        X['Fare'] = df['Fare'].fillna(df['Fare'].median()).astype(float)
        X['FamilySize'] = df['FamilySize'].astype(float)

        # Sex
        X['Sex_male'] = (df['Sex'] == 'male').astype(int)

        # Title encoding - keep main titles and Rare as group - map to categorical codes
        titles = ['Mr', 'Miss', 'Mrs', 'Master']
        X['Title'] = df['Title'].apply(lambda t: t if t in titles else 'Rare')
        X['Title_code'] = pd.Categorical(X['Title']).codes
        X = X.drop(columns=['Title'])

        # Deck code
        X['Deck_code'] = pd.Categorical(df['Deck']).codes

        # Embarked code
        X['Embarked_code'] = pd.Categorical(df['Embarked']).codes

        return X

    # Fit regressor on rows where Age is not null
    train_for_age = train[train['Age'].notnull()].copy()
    if train_for_age.shape[0] < 10:
        # fallback - not enough data
        median_age = train['Age'].median()
        train['Age'] = train['Age'].fillna(median_age)
        test['Age'] = test['Age'].fillna(median_age)
        return train, test

    X_age = prepare_for_regression(train_for_age)
    y_age = train_for_age['Age'].astype(float)

    # Regressor
    rfr = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rfr.fit(X_age, y_age)

    # Predict missing ages in train
    train_missing = train[train['Age'].isnull()]
    if not train_missing.empty:
        X_missing = prepare_for_regression(train_missing)
        train.loc[train['Age'].isnull(), 'Age'] = rfr.predict(X_missing)

    # Predict missing ages in test
    test_missing = test[test['Age'].isnull()]
    if not test_missing.empty:
        X_missing_test = prepare_for_regression(test_missing)
        test.loc[test['Age'].isnull(), 'Age'] = rfr.predict(X_missing_test)

    # As safety, fill any remaining nulls with median
    median_age = train['Age'].median()
    train['Age'] = train['Age'].fillna(median_age)
    test['Age'] = test['Age'].fillna(median_age)

    return train, test


def add_buckets(df: pd.DataFrame):
    df = df.copy()
    # Age bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,30,50,80], labels=['Child','Teen','YoungAdult','Adult','Senior'])
    # Fare bins (use quantiles)
    df['FareBin'] = pd.qcut(df['Fare'].fillna(0)+1, q=4, labels=['Low','MedLow','MedHigh','High'])
    # Interaction features
    df['Sex_Pclass'] = df['Sex'].astype(str) + "_" + df['Pclass'].astype(str)
    return df


# -------------------------
# Main pipeline
# -------------------------
def build_preprocessor(num_cols, cat_cols):
    """
    Build ColumnTransformer safely handling sklearn versions for OneHotEncoder param names.
    """
    # numeric pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # onehot encoder - support both sparse and sparse_output arguments
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe)
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')

    return preprocessor


def main(args):
    start_time = time.time()
    train_path = args.train or "train.csv"
    test_path = args.test or "test.csv"

    train = safe_read_csv(train_path)
    test = safe_read_csv(test_path)
    print(f"Loaded train {train.shape} / test {test.shape}")

    # Basic feature engineering (before age imputation)
    train_fe = feature_engineer_base(train)
    test_fe = feature_engineer_base(test)

    # Impute Age using model
    print("Imputing Age using RandomForestRegressor...")
    train_fe, test_fe = impute_age_model(train_fe, test_fe)

    # Further engineering
    train_fe = add_buckets(train_fe)
    test_fe = add_buckets(test_fe)

    # Choose features for modeling
    # We'll include numeric features and a selection of categorical features (one-hoted)
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'Title', 'FamilySize', 'IsAlone', 'FarePerPerson', 'Deck',
        'AgeBin', 'FareBin', 'Sex_Pclass'
    ]

    # Fill any remaining NaNs conservatively
    for df in [train_fe, test_fe]:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
        df['Deck'] = df['Deck'].fillna('U')
        df['Title'] = df['Title'].fillna('Rare')

    X = train_fe[features].copy()
    y = train_fe['Survived'].copy()
    X_test = test_fe[features].copy()

    # Define column lists for preprocessor
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FarePerPerson']
    # categorical columns to be one-hot encoded
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Deck', 'AgeBin', 'FareBin', 'Sex_Pclass']

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # split for local validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    print(f"Train/Val: {X_train.shape}/{X_val.shape}")

    # ---------- Model definitions ----------
    # Logistic Regression (with regularization tuning)
    pipe_lr = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, solver='saga', random_state=RANDOM_STATE))
    ])

    # Random Forest
    pipe_rf = Pipeline([
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    ])

    # XGBoost / LightGBM if available
    estimators_for_stack = []
    base_models = []

    # Logistic used as a base model as well
    estimators_for_stack.append(('lr', clone(pipe_lr)))
    base_models.append(('lr', clone(pipe_lr)))

    estimators_for_stack.append(('rf', clone(pipe_rf)))
    base_models.append(('rf', clone(pipe_rf)))

    if XGB_AVAILABLE:
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
        pipe_xgb = Pipeline([('preproc', preprocessor), ('clf', xgb_clf)])
        estimators_for_stack.append(('xgb', pipe_xgb))
        base_models.append(('xgb', pipe_xgb))
    else:
        print("XGBoost not available; skipping XGB model.")

    if LGBM_AVAILABLE:
        lgb_clf = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        pipe_lgb = Pipeline([('preproc', preprocessor), ('clf', lgb_clf)])
        estimators_for_stack.append(('lgbm', pipe_lgb))
        base_models.append(('lgbm', pipe_lgb))
    else:
        print("LightGBM not available; skipping LGBM model.")

    # Stacking classifier - final estimator
    final_estimator = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=RANDOM_STATE)
    stacking = StackingClassifier(estimators=estimators_for_stack, final_estimator=final_estimator, cv=5, n_jobs=-1, passthrough=False)

    # ---------- Quick evaluation of base models ----------
    print("\nTraining baseline models (quick fit on split)...")
    # fit logistic and RF baseline to get an initial sense
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_val)
    print("LR val acc:", round(accuracy_score(y_val, y_pred), 4))

    pipe_rf.fit(X_train, y_train)
    y_pred = pipe_rf.predict(X_val)
    print("RF val acc:", round(accuracy_score(y_val, y_pred), 4))

    if XGB_AVAILABLE:
        pipe_xgb.fit(X_train, y_train)
        print("XGB val acc:", round(accuracy_score(y_val, pipe_xgb.predict(X_val)), 4))
    if LGBM_AVAILABLE:
        pipe_lgb.fit(X_train, y_train)
        print("LGBM val acc:", round(accuracy_score(y_val, pipe_lgb.predict(X_val)), 4))

    # ---------- Hyperparameter tuning (optional, time-consuming) ----------
    if not args.fast:
        print("\nRunning RandomizedSearchCV for RandomForest (faster than grid)...")
        param_dist_rf = {
            'clf__n_estimators': [100, 200, 400],
            'clf__max_depth': [None, 6, 10, 15],
            'clf__min_samples_split': [2, 5, 8],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__max_features': ['auto', 'sqrt', 0.5]
        }
        rs_rf = RandomizedSearchCV(pipe_rf, param_distributions=param_dist_rf, n_iter=20,
                                   scoring='accuracy', cv=5, random_state=RANDOM_STATE, n_jobs=-1)
        rs_rf.fit(X, y)
        print("Best RF params:", rs_rf.best_params_)
        print("Best RF CV score:", round(rs_rf.best_score_, 4))

        # Replace RF in stacking and base_models with tuned version
        tuned_rf = rs_rf.best_estimator_
        # update lists
        # create new estimators_for_stack with tuned_rf replacing rf
        new_estimators = []
        for name, est in estimators_for_stack:
            if name == 'rf':
                new_estimators.append(('rf', tuned_rf))
            else:
                new_estimators.append((name, est))
        estimators_for_stack = new_estimators
        stacking = StackingClassifier(estimators=estimators_for_stack, final_estimator=final_estimator, cv=5, n_jobs=-1, passthrough=False)
    else:
        print("\n--fast mode: skipping heavy hyperparameter searches (use --fast to enable).")

    # ---------- Train final stacking model on full training data ----------
    print("\nTraining final stacking model on full training data (this may take a little while)...")
    stacking.fit(X, y)
    print("Stacking fitted. Evaluating with cross_val_score (5-fold)...")
    cv_scores = cross_val_score(stacking, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print("Stack CV scores:", np.round(cv_scores, 4), "mean:", round(np.mean(cv_scores), 4))

    # Holdout validation
    y_val_pred = stacking.predict(X_val)
    print("\nValidation results (stacking):")
    print("Accuracy:", round(accuracy_score(y_val, y_val_pred), 4))
    print(classification_report(y_val, y_val_pred))

    # ---------- Final training on full train then predict test ----------
    print("\nRetraining stacking on full training set and predicting test set...")
    stacking.fit(X, y)
    test_preds = stacking.predict(X_test).astype(int)

    submission = pd.DataFrame({
        "PassengerId": test_fe['PassengerId'],
        "Survived": test_preds
    })
    submission.to_csv(SUBMISSION_FN, index=False)
    print(f"Saved {SUBMISSION_FN} (shape {submission.shape})")

    # Save the stacking pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(stacking, FINAL_MODEL_FN)
    print(f"Saved final stacked model to {FINAL_MODEL_FN}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.2f} minutes")
    print("You can now upload submission.csv to Kaggle.")
    print("Screenshot you uploaded earlier (local path):")
    print("/mnt/data/Screenshot 2025-11-23 at 17.51.06.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Titanic pipeline with stacking and model-based age imputation.")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test.csv")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip heavy hyperparameter tuning")
    args = parser.parse_args()
    main(args)
