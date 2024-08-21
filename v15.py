# 0.8331

# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_selector
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# %%
X_train = pd.read_csv('./datasets/final_proj_data.csv')
X_test = pd.read_csv('./datasets/final_proj_test.csv')

y = X_train['y']
X_train = X_train.drop('y', axis=1)

# %%
X_train.info()

# %%
X_train.isna().sum()

# %%
missing_values = X_train.isna().sum()
threshold = 0.35
columns_to_drop = missing_values[missing_values > threshold * X_train.shape[0]].index

X_train.drop(columns=columns_to_drop, inplace=True)
X_test.drop(columns=columns_to_drop, inplace=True)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'),  make_column_selector(dtype_exclude='object')),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ('svd', TruncatedSVD(n_components=20))
        ]), make_column_selector(dtype_include='object'))
    ]
)

# %%
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('scaler', RobustScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# %%
scores = cross_val_score(pipeline, X_train, y, cv=5, scoring='balanced_accuracy')
print(f'Cross-validated balanced accuracy: {np.mean(scores):.4f}')

param_grid = {
    'classifier__n_estimators': [50, 100],  # Fewer options for number of trees
    'classifier__max_depth': [10, 20],  # Limit the depth options
    'classifier__min_samples_split': [2, 5],  # Reduce the number of split options
    'classifier__min_samples_leaf': [1, 2]  # Fewer options for minimum samples at leaf
}

grid_search = GridSearchCV(pipeline, 
                           param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           scoring='balanced_accuracy', 
                           verbose=2)

grid_search.fit(X_train, y)

print(f'Best cross-validated balanced accuracy: {grid_search.best_score_:.4f}')

# %%
client_ids = X_test.index 
results = pd.DataFrame({
    'index': client_ids,
    'y': grid_search.best_estimator_.predict(X_test)
})
results.to_csv('./datasets/submission.csv', index=False)

# -*- coding: utf-8 -*-

