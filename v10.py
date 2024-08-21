# 0.7955

# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.decomposition import PCA, TruncatedSVD
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score
from category_encoders.woe import WOEEncoder
from sklearn.linear_model import LogisticRegression

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
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(exclude=['object']).columns

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ('svd', TruncatedSVD(n_components=20))
        ]), categorical_features)
    ]
)

# %%
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    # ('pca', PCA(n_components=20)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# %%
scores = cross_val_score(pipeline, X_train, y, cv=5, scoring='balanced_accuracy')
print(f'Cross-validated balanced accuracy: {np.mean(scores):.4f}')


param_grid = {
    'classifier__C': [0.1, 1, 10],   # Regularization strength
    'classifier__penalty': ['l2'],  # Type of regularization
    'classifier__solver': ['liblinear'],  # Solver to use for optimization
    'classifier__max_iter': [100, 200, 500]  # Maximum number of iterations
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

