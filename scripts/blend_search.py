#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score

# ----------------------------
# Load and prepare the data
# ----------------------------
train = pd.read_csv("train.csv")

def extract_title(name):
    try:
        return name.split(",")[1].split(".")[0].strip()
    except:
        return "Unknown"

# Conservative Titanic features
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Title'] = train['Name'].apply(extract_title).replace(['Mlle','Ms'],'Miss').replace({'Mme':'Mrs'})
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
train['FarePerPerson'] = train['Fare'] / train['FamilySize']
train['Deck'] = train['Cabin'].fillna('U').apply(lambda x: x[0] if isinstance(x,str) and len(x) else 'U')
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().iloc[0])

features = [
    'Pclass','Sex','Age','SibSp','Parch','Fare',
    'Embarked','Title','FamilySize','IsAlone','FarePerPerson','Deck'
]

X = train[features]
y = train['Survived']

# ----------------------------
# Preprocessor
# ----------------------------
num_cols = ['Age','SibSp','Parch','Fare','FamilySize','FarePerPerson']
cat_cols = ['Pclass','Sex','Embarked','Title','IsAlone','Deck']

# Fix for sklearn 1.7+
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except:
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
])

# ----------------------------
# OOF function
# ----------------------------
def get_oof_probs(pipe, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for tr, val in skf.split(X, y):
        p = clone(pipe)
        p.fit(X.iloc[tr], y.iloc[tr])
        oof[val] = p.predict_proba(X.iloc[val])[:, 1]

    return oof

# ----------------------------
# Build models
# ----------------------------
pipe_rf = Pipeline([
    ('pre', pre),
    ('clf', RandomForestClassifier(n_estimators=400, max_features='sqrt', random_state=42))
])

pipe_gb = Pipeline([
    ('pre', pre),
    ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42))
])

print("Generating OOF predictions...")
oof_rf = get_oof_probs(pipe_rf, X, y)
oof_gb = get_oof_probs(pipe_gb, X, y)

# ----------------------------
# Blend search
# ----------------------------
best_score = 0
best_weight = None

print("Searching best RF/GB blend weight...")

for w in np.linspace(0, 1, 21):  # 0.0 to 1.0 step 0.05
    preds = (w * oof_rf + (1 - w) * oof_gb) >= 0.5
    acc = accuracy_score(y, preds)

    if acc > best_score:
        best_score = acc
        best_weight = w

print("\nBest blend:")
print(f"RF weight = {best_weight:.2f}, GB weight = {1-best_weight:.2f}")
print(f"CV accuracy = {best_score:.4f}")
