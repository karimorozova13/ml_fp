#  0.7587

# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD

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
threshold = 0.4
columns_to_drop = missing_values[missing_values > threshold * X_train.shape[0]].index

X_train.drop(columns=columns_to_drop, inplace=True)
X_test.drop(columns=columns_to_drop, inplace=True)

# %%
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(exclude=['object']).columns

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False))
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# %%
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('svd', TruncatedSVD(n_components=20, random_state=42)),
    # ('pca', PCA(n_components=20)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# %%
scores = cross_val_score(pipeline, X_train, y, cv=5, scoring='balanced_accuracy')
print(f'Cross-validated balanced accuracy: {np.mean(scores):.4f}')


param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(pipeline, 
                           param_grid, 
                           cv=5, n_jobs=-1, 
                           scoring='balanced_accuracy', 
                           verbose=2)

grid_search.fit(X_train, y)

print(f'Best cross-validated balanced accuracy: {grid_search.best_score_:.4f}')

# %%
X_test['y'] = grid_search.best_estimator_.predict(X_test)

# %%
X_test.columns

# %%
client_ids = X_test.index 
results = pd.DataFrame({
    'index': client_ids,
    'y': X_test['y']
})
results.to_csv('./datasets/submission.csv', index=False)
