# import numpy as np

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import PowerTransformer

# from sklearn.compose import ColumnTransformer

# from sklearn.impute import SimpleImputer

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# import pandas as pd
# import matplotlib.pyplot as plt

# from joblib import dump


# numerical = [
#     'avg_glucose_level',
#     'bmi',
#     'age'
# ]

# categorical = [
#     'gender',
#     'hypertension',
#     'heart_disease',
#     'ever_married',
#     'work_type',
#     'Residence_type',
#     'smoking_status'
# ]


# num_pipeline = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('power', PowerTransformer(
#         method='yeo-johnson',
#         standardize=True
#     ))
# ])


# cat_pipeline = Pipeline(steps=[
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))
# ])


# transformer = ColumnTransformer(transformers=[
#     ('num', num_pipeline, numerical),
#     ('cat', cat_pipeline, categorical)
# ])


# def get_models():

#     models = []
#     names = []

#     models.append(
#         LogisticRegression(
#             solver='liblinear',
#             random_state=42
#         )
#     )
#     names.append('Logistic Regression')

#     models.append(
#         LinearDiscriminantAnalysis()
#     )
#     names.append('LDA')

#     models.append(
#         RandomForestClassifier(
#             n_estimators=100,
#             random_state=42
#         )
#     )
#     names.append('Random Forest')

#     return models, names


# def evaluate_model(x, y, model):

#     cv = RepeatedStratifiedKFold(
#         n_splits=10,
#         n_repeats=3,
#         random_state=42
#     )

#     scores = cross_val_score(
#         model,
#         x,
#         y,
#         scoring='roc_auc',
#         cv=cv,
#         n_jobs=-1
#     )

#     return scores


# x = df.drop('stroke', axis=1)

# y = df['stroke']


# models, names = get_models()

# results = []

# for i in range(len(models)):

#     pipeline = Pipeline(steps=[
#         ('transformer', transformer),
#         ('smote', SMOTE(random_state=42)),
#         ('model', models[i])
#     ])

#     scores = evaluate_model(x, y, pipeline)

#     results.append(scores)

#     print(
#         f">{names[i]}: "
#         f"ROC-AUC = {np.mean(scores):.3f} "
#         f"({np.std(scores):.3f})"
#     )

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('./stroke-data.csv')
    df = df.drop('id', axis=1)
    categorical = ['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical = ['avg_glucose_level', 'bmi', 'age']
    y = df['stroke']
    X = df.drop('stroke', axis=1)
    return X, y, categorical, numerical

def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return scores

# Load data
X, y, categorical, numerical = load_data()
print(X.shape, y.shape)

# Define the LDA model
model = LinearDiscriminantAnalysis()

# Prepare the pipeline
transformer = ColumnTransformer(transformers=[
    ('imp', SimpleImputer(strategy='median'), numerical),
    ('o', OneHotEncoder(handle_unknown='ignore'), categorical)  # handle_unknown='ignore' to manage unseen categories
])

pipeline = Pipeline(steps=[
    ('t', transformer),
    ('p', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('over', SMOTE()),
    ('m', model)
])

# Evaluate the model
scores = evaluate_model(X, y, pipeline)
# print('LDA %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Plot the results
plt.boxplot([scores], labels=['LDA'], showmeans=True)
plt.show()

# Fit the pipeline on the entire dataset
pipeline.fit(X, y)

# Save the trained pipeline
dump(pipeline, 'stroke_prediction_model.joblib')