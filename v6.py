import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Завантажте дані
X_train = pd.read_csv('./datasets/final_proj_data.csv')
X_test = pd.read_csv('./datasets/final_proj_test.csv')

# Виберіть цільову змінну
y = X_train['y']
X_train = X_train.drop('y', axis=1)

# Розділіть дані на тренувальний і валідаційний набори
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y, test_size=0.2, random_state=42)

# %%
from sklearn.impute import SimpleImputer
num_features = X_train_split.select_dtypes(include=['float64', 'int64']).columns
cat_features = X_train_split.select_dtypes(include=['object']).columns
# Створіть імпутер для числових ознак
num_imputer = SimpleImputer(strategy='median')

# Створіть імпутер для категоріальних ознак
cat_imputer = SimpleImputer(strategy='most_frequent')

# Обробка числових ознак
X_train_split[num_features] = num_imputer.fit_transform(X_train_split[num_features])
X_val_split[num_features] = num_imputer.transform(X_val_split[num_features])

# Обробка категоріальних ознак
X_train_split[cat_features] = cat_imputer.fit_transform(X_train_split[cat_features])
X_val_split[cat_features] = cat_imputer.transform(X_val_split[cat_features])

# %%
# Визначте числові та категоріальні ознаки


# Створіть конвеєр для обробки числових та категоріальних ознак
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# Створіть конвеєр для балансу класів та моделі
pipeline = imPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Навчіть модель на збалансованих даних
pipeline.fit(X_train_split, y_train_split)

# Оцінка моделі
from sklearn.metrics import classification_report

# Прогнозування на валідаційному наборі
y_pred = pipeline.predict(X_val_split)

# Оцінка моделі
print(classification_report(y_val_split, y_pred))

# Навчання моделі на всіх даних
pipeline.fit(X_train, y)

# Прогнозування на тестовому наборі
y_test_pred = pipeline.predict(X_test)

# Формування файлу submission.csv
client_ids = X_test.index
results = pd.DataFrame({
    'index': client_ids,
    'y': y_test_pred
})
results.to_csv('./datasets/submission.csv', index=False)
