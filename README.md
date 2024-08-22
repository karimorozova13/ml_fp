### README 

This project focuses on building and evaluating a machine learning model to predict outcomes based on the provided dataset. 
The process involves data preprocessing, pipeline construction, hyperparameter tuning, and model evaluation.

### Step-by-Step Description:

**Step 1: Importing Necessary Packages**  
The first step involves importing all necessary Python packages that will be used for data processing, model building, 
and evaluation.

**Step 2: Loading and Preparing the Dataset**  
The training and test datasets are loaded from the `final_proj_data.csv` and `final_proj_test.csv` files.

**Step 3: Separating the Target Variable from the Features**  
The target variable `y` is separated from the dataset, and all further operations are performed on the feature set.

**Step 4: Data Structure Overview**  
The structure of the dataset is reviewed to understand the types of variables.

**Step 5: Checking for Missing Values**  
Columns with missing values are checked to determine which have a significant number of missing entries.

**Step 6: Removing Columns with a High Number of Missing Values**  
Columns with a high number of missing values are dropped from both the training 
and test datasets based on a specified threshold ( > 40%).

**Step 7: Defining Preprocessing Steps for Numerical and Categorical Features**  
A ColumnTransformer is created to perform the following operations:
- Imputation of missing values in numerical columns using the median.
- Imputation of missing values in categorical columns using the most frequent value.
- One-hot encoding of categorical features.
- Dimensionality reduction of encoded features using Truncated SVD.

**Step 8: Building a Pipeline for Data Preprocessing, Resampling, and Model Training**  
A pipeline is constructed to perform the following steps:
- Apply preprocessing.
- Apply SMOTE to address class imbalance.
- Scale features.
- Train a Random Forest classifier.

**Step 9: Evaluating the Pipeline with Cross-Validation**  
The pipeline is evaluated using cross-validation to determine its accuracy.

**Step 10: Defining a Parameter Grid for Hyperparameter Tuning with GridSearchCV**  
A parameter grid is defined for the classifier, including:
- The number of trees in the forest.
- The maximum depth of the trees.
- The minimum number of samples required to split a node.
- The minimum number of samples required at a leaf node.

**Step 11: Performing Grid Search to Find the Best Model Parameters**  
GridSearchCV is used to search for the best hyperparameters for the model based on the defined parameter grid.

**Step 12: Outputting the Best Cross-Validated Balanced Accuracy**  
The best cross-validated balanced accuracy obtained during the grid search is output.

**Step 13: Generating Predictions on the Test Set and Saving the Results**  
Predictions are generated on the test set, and the results are saved in a file named `submission.csv`.

### Conclusions:

1. Using other categorical encoders, such as TargetEncoder and WOEEncoder, resulted in worse prediction performance.

2. Attempts to build an ensemble model using stacking did not achieve the desired accuracy.

This approach provided a clear and consistent method for building a robust machine learning model, 
with a focus on data preprocessing, class balancing, and hyperparameter optimization.