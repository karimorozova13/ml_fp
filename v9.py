import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data
X_train = pd.read_csv('./datasets/final_proj_data.csv')
X_test = pd.read_csv('./datasets/final_proj_test.csv')

y = X_train['y']
X_train = X_train.drop('y', axis=1)

# Preprocessor: Imputation and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X_train.select_dtypes(exclude=['object']).columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', TargetEncoder())
        ]), X_train.select_dtypes(include=['object']).columns)
    ]
)

# Define the base models for stacking
base_models = [
    ('gbc', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

# Meta-model for stacking
meta_model = LogisticRegression()

# Define the stacking model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

# Create the main pipeline with preprocessor, SMOTE, and stacking model
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('stacking', stacking_model)
])

# K-Fold Cross-Validation with 10 folds
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_val_score(pipeline, X_train, y, cv=kfold, scoring='balanced_accuracy', n_jobs=-1)

print(f'10-Fold Cross-validated balanced accuracy: {np.mean(cv_results):.4f}')

# Grid Search to optimize the stacking model
param_grid = {
    'stacking__gbc__n_estimators': [50, 100, 200],
    'stacking__rf__n_estimators': [50, 100, 200],
    'stacking__final_estimator__C': [0.1, 1.0, 10.0]  # Tuning regularization parameter of the meta-model
}

grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='balanced_accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y)

print(f'Best cross-validated balanced accuracy after Grid Search: {grid_search.best_score_:.4f}')

# Predict on test data
client_ids = X_test.index 
results = pd.DataFrame({
    'index': client_ids,
    'y': grid_search.best_estimator_.predict(X_test)
})
results.to_csv('./datasets/submission.csv', index=False)


