import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('Train.csv')

# Step 1: Basic EDA
print(data.head())
print(f"Shape of the dataset: {data.shape}")
print(f"Columns in the dataset: {data.columns.tolist()}")
print(data.info())
print(data.describe())

# Step 2: Analyze categorical columns and target column
def analyze_column(col):
    print(f"\nUnique values in '{col}':")
    print(data[col].value_counts())

analyze_column('Crop')
analyze_column('category')

# Step 3: Encode target column (multi-class classification)
le = LabelEncoder()
data['category'] = le.fit_transform(data['category'])
print("Encoded target classes:", dict(zip(le.classes_, le.transform(le.classes_))))

# Step 4: Date processing
data['SDate'] = pd.to_datetime(data['SDate'])
data['HDate'] = pd.to_datetime(data['HDate'])
data['Duration'] = (data['HDate'] - data['SDate']).dt.days
data['SDate_month'] = data['SDate'].dt.month
data['HDate_month'] = data['HDate'].dt.month
data.drop(['SDate', 'HDate'], axis=1, inplace=True)

# Step 5: Handle missing values
numerical_columns = ['CropCoveredArea', 'CHeight', 'IrriCount', 'WaterCov', 'ExpYield', 'Duration']
missing_values = data.isnull().sum()
print("\nMissing values:")
print(missing_values)
for col in missing_values[missing_values > 0].index:
    if col in numerical_columns:
        data[col].fillna(data[col].mean(), inplace=True)  # Impute numerical columns with mean
    else:
        data[col].fillna(data[col].mode()[0], inplace=True)  # Impute categorical columns with mode

# Step 6: Detect and remove duplicates
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
data.drop_duplicates(inplace=True)

# Step 7: Analyze and handle outliers
for col in numerical_columns:
    sns.boxplot(x=data[col])
    plt.title(f'Outliers in {col}')
    plt.show()

for col in numerical_columns:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    capped_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
    print(f"Capped {capped_count} outliers in {col}")
    data[col] = np.clip(data[col], lower_bound, upper_bound)

# Step 8: Rename columns
data.rename(columns={'CTransp': 'PlantingMethod'}, inplace=True)

# Step 9: Drop irrelevant columns
data = data.drop([ 'District', 'Sub-District',
                   'FarmID', 'CNext', 'geometry'], axis=1)
print(f"Columns in the dataset: {data.columns.tolist()}")

# Step 10: Target encoding for categorical columns

columns_to_encode = data.select_dtypes(include=['object']).columns

# Apply one-hot encoding to the remaining categorical columns
data_encoded = pd.get_dummies(data, columns=columns_to_encode)

print(data_encoded.head())

for col in columns_to_encode:
    print(f"{col}: {data[col].nunique()} unique values")

bool_columns = data_encoded.select_dtypes(include=[bool]).columns

# Identify the boolean columns (those that are True/False) in the encoded data
bool_columns = data_encoded.select_dtypes(include=[bool]).columns

# Convert the boolean columns to integers (1 for True, 0 for False)
data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

# Verify the changes
print(data_encoded.head())

# Save the updated, not encoded, dataset to a new CSV file
data.to_csv('Train_processed.csv', index=False)

df = data_encoded
print(df.shape)

# Step 11: Feature selection
X = df.drop(columns=['category'])
y = df['category']

# Mutual Information
mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Mutual Information Scores:")
print(mi_scores)

# Correlation analysis
correlation_matrix = df[numerical_columns + ['category']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Recursive Feature Elimination
model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=30)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print("Selected Features by RFE:", selected_features.tolist())

# Final feature set
final_features = list(set(selected_features) | set(mi_scores.head(20).index))
X = X[final_features]
print(X.columns, "\n")

# Step 12: Train-validation split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)


print("Columns in X_val:", X_val.columns)



# Step 14: Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Step 13: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Visualize Class Distribution Before and After SMOTE
plt.figure(figsize=(8, 5))
y_train_original = y_temp[y_temp.index.isin(X_train.index)]  # Original y_train before SMOTE
y_train_original.value_counts().plot(kind='bar', color='blue', alpha=0.7)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
pd.Series(y_train).value_counts().plot(kind='bar', color='green', alpha=0.7)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show()

# Step 14: Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train['category'] = y_train
X_val['category'] = y_val
X_test['category'] = y_test

X_train.to_csv('processed_train.csv', index=False)
X_val.to_csv('processed_val.csv', index=False)
X_test.to_csv('processed_test.csv', index=False)