import pandas as pd
import matplotlib.pyplot as plt # Used to make a histogram for TASK 2.1 
import numpy as np # Used to convert ? to NaN for TASK 3

# TASK 1

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Define column names
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

# Load the dataset
data = pd.read_csv(url, names=column_names, skipinitialspace=True)


# print(data.head()) # print first five lines of dataframe
print(data) # print the whole dataframe (with limiations as dataset is too large).


#  TASK 2
print("\nTASK 2\n")

# Display the first five rows
print("\n\nFirst few rows:")
print(data.head())

# DataFrame information
print("\nDataFrame Info:")
print(data.info())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Shape of the DataFrame
print("\nShape of the DataFrame:")
print(data.shape)

# TASK 2.1
# print("\n\nTASK 2.1 Histogram: \n(a pop-up should appear)")

# data['age'].plot.hist(bins=50)  # Adjust the number of bins as needed
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()


# TASK 3
print("\nTASK 3\n")

# Convert '?' to NaN to make it easier to count missing values
data.replace('?', np.nan, inplace=True)

# Now we can count the number of NaN values in each column
missing_values_count = data.isna().sum()

# Display the number of missing values for each column
print(missing_values_count)


# TASK 4
print("\nTASK 4\n")

# Check for duplicate rows
duplicate_rows = data.duplicated()

# Count the number of duplicate rows
num_duplicate_rows = duplicate_rows.sum()

# Remove duplicate rows
cleaned_data = data.drop_duplicates()
data = cleaned_data

# Print the number of duplicate rows found
print("Number of duplicate rows:", num_duplicate_rows)


## TASK 5
print("\nTASK 5\n")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset (assuming 'data' has already been loaded with the correct columns)
# data = pd.read_csv(url, names=column_names, skipinitialspace=True)

# Handling the target column (assuming 'income' is the target)
target = 'income'

# Separating the features from the target
X = data.drop(columns=[target])
y = data[target]

# Identifying numerical and categorical columns (excluding the target column)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Pipeline for numerical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combining pipelines with ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Applying the preprocessing pipeline
X_preprocessed = preprocessor.fit_transform(X)

# Convert the preprocessed data back into a DataFrame for easier handling
columns_transformed = preprocessor.get_feature_names_out()
X_preprocessed_df = pd.DataFrame(X_preprocessed.toarray(), columns=columns_transformed)

# Reattach the target column to meet the requirement of having 106 columns
X_preprocessed_with_target = pd.concat([X_preprocessed_df, y.reset_index(drop=True)], axis=1)

# Printing the pipeline for display
print(preprocessor)

# Print the dataframe's shape after preprocessing and including the target column
print('Dataframe shape after preprocessing and including the target column:', X_preprocessed_with_target.shape)


# TASK 6
print("\nTASK 6\n")

# Print out initial value counts for the 'income' column to check for inconsistencies
print(data['income'].value_counts())

# Clean up inconsistencies in the 'income' column if necessary
data['income'] = data['income'].str.replace('.', '')

# Verify the cleanup by printing the value counts again
print(data['income'].value_counts())


# TASK 7
print("\nTASK 7\n")

from sklearn.model_selection import train_test_split

# Assuming 'X_preprocessed_with_target' is the DataFrame after preprocessing which includes the target column
# First, we separate features and the target variable again
X_final = X_preprocessed_with_target.drop(columns=['income'])
y_final = X_preprocessed_with_target['income']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.20, random_state=42)

# Print the shapes of the train/test sets in one command
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# TASK 8
print("\nTASK 8\n")

from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# List of C values to iterate over
C_values = [0.1, 1, 10]

# Train, predict, and evaluate for each C value
for C in C_values:
    print(f"\nTraining with C = {C}\n")
    svm_clf = SVC(kernel='rbf', gamma=1, C=C)
    svm_clf.fit(X_train, y_train)
    
    # TASK 8.1: Test your model and report
    y_predict = svm_clf.predict(X_test)
    print(f"Classification Report for C = {C}:\n")
    print(classification_report(y_test, y_predict))
    
    # TASK 8.2: Plot the confusion matrix
    print(f"Confusion Matrix for C = {C}:\n")
    ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    plt.title(f'Confusion Matrix (C = {C})')
    plt.show()
