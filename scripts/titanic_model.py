#!/usr/bin/env python3
"""
titanic_model.py

End-to-end Titanic ML script (no Jupyter required).
- Loads train.csv and test.csv (expected in same folder)
- Feature engineering
- Preprocessing pipelines
- Trains Logistic Regression and Random Forest
- Simple GridSearch for Random Forest
- Trains final model on full training set
- Writes submission.csv (PassengerId, Survived)
- Saves final pipeline+model using joblib

Usage:
    python titanic_model.py
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import joblib

# -------------------------
# Configuration / Defaults
# -------------------------
RANDOM_STATE = 42
MODEL_DIR = "models"
SUBMISSION_FN = "submission.csv"
FINAL_MODEL_FN = os.path.join(MODEL_DIR, "final_random_forest.joblib")

# -------------------------
# Utility & feature functions
# -------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with engineered features."""
    df = df.copy()
    # Title from Name
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Fare per person
    # Guard against division by zero (shouldn't happen because FamilySize >= 1)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Deck from Cabin (first character). Use 'U' if missing
    df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0] if (isinstance(x, str) and len(x) > 0) else 'U')

    return df

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Required file not found: {path}")
        sys.exit(1)
    return pd.read_csv(path)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Main flow
# -------------------------
def main(args):
    # 1) Load data
    train_path = args.train if args.train else "train.csv"
    test_path = args.test if args.test else "test.csv"

    print("Loading data...")
    train = safe_read_csv(train_path)
    test = safe_read_csv(test_path)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Basic sanity
    if 'Survived' not in train.columns:
        print("[ERROR] train.csv must contain 'Survived' column.")
        sys.exit(1)

    # 2) Feature engineering
    print("Feature engineering...")
    train_fe = feature_engineer(train)
    test_fe = feature_engineer(test)

    # 3) Select features used for modeling
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'Title', 'FamilySize', 'IsAlone', 'FarePerPerson', 'Deck'
    ]
    X = train_fe[features].copy()
    y = train_fe['Survived'].copy()
    X_test = test_fe[features].copy()

    # 4) Define numeric and categorical columns
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FarePerPerson']
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Deck']

    # 5) Preprocessing pipelines
    print("Building preprocessing pipelines...")
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')

    # 6) Local train/val split for quick evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=RANDOM_STATE, stratify=y)
    print(f"Train/Val split: {X_train.shape} / {X_val.shape}")

    # 7) Logistic Regression baseline
    print("\nTraining Logistic Regression baseline...")
    pipe_log = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear', random_state=RANDOM_STATE))
    ])
    pipe_log.fit(X_train, y_train)
    y_pred_log = pipe_log.predict(X_val)
    acc_log = accuracy_score(y_val, y_pred_log)
    print(f"Logistic Regression accuracy (val): {acc_log:.4f}")
    print(classification_report(y_val, y_pred_log))

    # 8) Random Forest baseline
    print("\nTraining Random Forest baseline...")
    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)
    print(f"Random Forest accuracy (val): {acc_rf:.4f}")
    print(classification_report(y_val, y_pred_rf))

    # 9) Simple GridSearchCV for Random Forest on the FULL training dataset (best practice: use CV)
    print("\nRunning GridSearchCV to tune Random Forest (this may take a few minutes)...")
    param_grid_rf = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 6, 10],
        'clf__min_samples_split': [2, 5]
    }
    gs_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5,
                         scoring='accuracy', n_jobs=-1, verbose=0)
    gs_rf.fit(X, y)  # use whole train set for CV
    print("Best RF params:", gs_rf.best_params_)
    print(f"Best RF CV score: {gs_rf.best_score_:.4f}")

    # 10) Evaluate best model on the held-out validation set (if you kept X_val)
    best_rf = gs_rf.best_estimator_
    # Fit best_rf on X_train (we'll evaluate on X_val)
    best_rf.fit(X_train, y_train)
    y_pred_best_rf = best_rf.predict(X_val)
    acc_best_rf = accuracy_score(y_val, y_pred_best_rf)
    print(f"Best RF accuracy (val): {acc_best_rf:.4f}")
    print(classification_report(y_val, y_pred_best_rf))

    # Optional: CV scores (more robust)
    print("Running cross_val_score (5-fold) on best RF for a final estimate...")
    cv_scores = cross_val_score(gs_rf.best_estimator_, X, y, cv=5, n_jobs=-1)
    print("RF CV scores:", np.round(cv_scores, 4), "mean:", np.round(np.mean(cv_scores), 4))

    # 11) Train final model on full training data, then predict on test set
    print("\nTraining final model on full training data and creating submission...")
    final_model = gs_rf.best_estimator_
    final_model.fit(X, y)

    # Predict on test set
    test_preds = final_model.predict(X_test).astype(int)

    # Build submission DataFrame and save
    submission = pd.DataFrame({
        "PassengerId": test['PassengerId'],
        "Survived": test_preds
    })
    submission.to_csv(SUBMISSION_FN, index=False)
    print(f"Saved submission file: {SUBMISSION_FN} (shape: {submission.shape})")

    # 12) Save model pipeline
    print("Saving trained model pipeline...")
    ensure_dir(MODEL_DIR)
    joblib.dump(final_model, FINAL_MODEL_FN)
    print(f"Saved model pipeline to: {FINAL_MODEL_FN}")

    # 13) (Optional) Show top feature importances if RandomForest supports it
    try:
        rf_clf = final_model.named_steps['clf']
        # Get processed feature names
        # Handle onehot naming depending on sklearn version
        processed_feature_names = []
        # numeric names
        processed_feature_names.extend(num_cols)
        # onehot names
        cat_transformer = final_model.named_steps['preprocessor'].transformers_[1][1]
        ohe = None
        if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
            ohe = cat_transformer.named_steps['onehot']
        elif hasattr(cat_transformer, 'named_transformer'):  # unlikely path, safety
            ohe = cat_transformer.named_transformer
        if ohe is not None and hasattr(ohe, 'get_feature_names_out'):
            onehot_names = list(ohe.get_feature_names_out(cat_cols))
            processed_feature_names.extend(onehot_names)
        else:
            # Fallback: create simple placeholders for categorical features
            processed_feature_names.extend([f"cat_{c}" for c in cat_cols])

        importances = rf_clf.feature_importances_
        if len(importances) == len(processed_feature_names):
            feat_imp = pd.Series(importances, index=processed_feature_names).sort_values(ascending=False).head(20)
            print("\nTop feature importances (Random Forest):")
            print(feat_imp.to_string())
        else:
            print("\nWarning: Could not align feature importances with names (skipping explicit list).")
    except Exception as e:
        print("Could not compute feature importance:", e)

    print("\nAll done! You can now upload 'submission.csv' to the Kaggle Titanic competition page.")
    print("Example: kaggle competitions submit -c titanic -f submission.csv -m 'My RF submission' (if using kaggle-cli).")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic ML script (Logistic Regression + Random Forest).")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test.csv")
    parser.add_argument("--plots", action="store_true", help="Show plots (not recommended for headless runs)")
    args = parser.parse_args()
    main(args)
