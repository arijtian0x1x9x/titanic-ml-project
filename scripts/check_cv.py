# check_cv.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

train = pd.read_csv("train.csv")

# --- conservative features & preprocessing (same as optimal script) ---
def extract_title(name):
    try: return name.split(",")[1].split(".")[0].strip()
    except: return "Unknown"

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Title'] = train['Name'].apply(extract_title).replace(['Mlle','Ms'],'Miss').replace({'Mme':'Mrs'})
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
train['FarePerPerson'] = train['Fare'] / train['FamilySize']
train['Deck'] = train['Cabin'].fillna('U').apply(lambda x: x[0] if isinstance(x,str) and len(x) else 'U')
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().iloc[0])

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone','FarePerPerson','Deck']
X = train[features]
y = train['Survived']

num_cols = ['Age','SibSp','Parch','Fare','FamilySize','FarePerPerson']
cat_cols = ['Pclass','Sex','Embarked','Title','IsAlone','Deck']

# safe OHE
from sklearn.preprocessing import OneHotEncoder
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', ohe)])
from sklearn.compose import ColumnTransformer
pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')

# Choose model to evaluate (change)
model = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=400, max_features='sqrt', random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("CV scores:", scores, "mean:", scores.mean())
