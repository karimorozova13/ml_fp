# 0.5031

# %%
import pandas as pd

X_train  = pd.read_csv('./datasets/final_proj_data.csv')
X_test = pd.read_csv('./datasets/final_proj_test.csv')

y = X_train['y']
X_train = X_train.drop('y', axis=1)

# %%
X_train.info()
X_train.describe()

# %%
X_train.dtypes

X_train.isna().sum()

data = X_train.isna().sum().gt(9000)

# %%
columns_to_remove = data[data].index
X_train = X_train.drop(columns=columns_to_remove)
X_test = X_test.drop(columns=columns_to_remove)

# %%
X_train_cat = X_train.select_dtypes(include='object')
X_train_num = X_train.select_dtypes(exclude='object')

X_test_cat = X_test.select_dtypes(include='object')
X_test_num = X_test.select_dtypes(exclude='object')

# %%
from sklearn.impute import SimpleImputer

cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
num_imputer = SimpleImputer().set_output(transform='pandas')

X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.fit_transform(X_test_cat)

X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = cat_imputer.fit_transform(X_test_num)

# %%
X_test_num.isna().sum()

# %%
unique_counts = X_train_cat.nunique()
print(unique_counts)

# %%
from sklearn.preprocessing import TargetEncoder

encoder = TargetEncoder(random_state=42).set_output(transform='pandas')

X_train_cat = encoder.fit_transform(X_train_cat, y)
X_test_cat = encoder.transform(X_test_cat)

# %%
# normalize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas')

X_train_cat = scaler.fit_transform(X_train_cat)
X_test_cat = scaler.transform(X_test_cat)

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%
#check class' balance

y.value_counts(normalize=True)

# %%
X_train_cat.isna().sum()

# %%

X_train_concat = pd.concat([X_train_num,X_train_cat], axis=1)
X_test_concat = pd.concat([X_test_num, X_test_cat], axis=1)

# %%
X_train_concat.isna().sum()

# %%
# balanced the data
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X_train_concat, y)

# %%
from sklearn.decomposition import PCA

pca = PCA().set_output(transform='pandas').fit(X_res)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme()

explained_variance = np.cumsum(pca.explained_variance_ratio_)

ax = sns.lineplot(explained_variance)

ax.set(xlabel='number of components', ylabel='comulative explained variance')

n_components = np.searchsorted(explained_variance, 0.85)

ax.axvline(x=n_components,
           c='teal',
           linestyle='--',
           linewidth=0.75)

plt.show()

# %%
X_train_pca= pca.transform(X_res)
X_test_pca = pca.transform(X_test_concat)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=2)

grid_search.fit(X_res, y_res)

# %%
client_ids = X_test.index 
results = pd.DataFrame({
    'index': client_ids,
    'y': grid_search.best_estimator_.predict(X_test_concat)
})
results.to_csv('./datasets/submission.csv', index=False)

print(results.head())

