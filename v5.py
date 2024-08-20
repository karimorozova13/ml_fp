#  

# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_selector
from imblearn.pipeline import Pipeline as ImbPipeline

# %%
X_train = pd.read_csv('./datasets/final_proj_data.csv')
X_test = pd.read_csv('./datasets/final_proj_test.csv')

y = X_train['y']
X_train = X_train.drop('y', axis=1)

# %%
X_train.info()

# %%
missing_values = X_train.isna().sum()
threshold = 0.4
columns_to_drop = missing_values[missing_values > threshold * X_train.shape[0]].index

X_train.drop(columns=columns_to_drop, inplace=True)
X_test.drop(columns=columns_to_drop, inplace=True)

# %%

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder())])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer())])

# %%

preprocessor = (ColumnTransformer(
    transformers=[
        ('cat',
         cat_transformer,
         make_column_selector(dtype_include=object)),
        ('num',
         num_transformer,
         make_column_selector(dtype_include=np.number))],
    n_jobs=-1,
    verbose_feature_names_out=False).set_output(transform='pandas'))

# %%

model_pipeline = ImbPipeline(steps=[
    ('pre_processor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=20)),
    ('clf_estimator', GradientBoostingClassifier(random_state=42))])    

# %%
param_grid = {
   'clf_estimator__n_estimators': [100, 200],
    'clf_estimator__max_depth': [3, 6],
    'clf_estimator__learning_rate': [0.1, 0.2],
    'clf_estimator__subsample': [1.0],
    'clf_estimator__max_features': ['sqrt']
}

grid_search = GridSearchCV(model_pipeline, 
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


