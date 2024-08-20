import pandas as pd
import numpy as np
import re

def detect_issues(df):
    issues = []

    # Check for non-numeric characters in numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.contains('[^0-9\.\,\-]', regex=True).any():
                issues.append(f"Non-numeric characters found in numeric column '{col}'")

    # Check for columns with only one unique value (potential redundancy)
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"Column '{col}' has only one unique value, might be redundant")

    # Check for mixed data types in columns
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (int, float, str, np.datetime64))).nunique() > 1:
            issues.append(f"Mixed data types found in column '{col}'")

    # Check for missing values
    if df.isnull().values.any():
        issues.append("Missing values detected in the dataset")

    # Check for duplicate rows
    if df.duplicated().any():
        issues.append("Duplicate rows detected in the dataset")

    # Check for inconsistent formatting (e.g., different units)
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains(r'\b(sqft|sqm|Lac|₹|per sqft)\b', regex=True).any():
            issues.append(f"Inconsistent formatting detected in column '{col}'")

    # Check for high cardinality in categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            issues.append(f"High cardinality detected in column '{col}'")

    # Check for potential outliers in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].max() / df[col].min() > 1000:  # Example threshold
            issues.append(f"Potential outliers detected in column '{col}'")

    # Check for incorrect data types (e.g., dates stored as strings)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                pd.to_datetime(df[col])
                issues.append(f"Column '{col}' might be a date stored as a string")
            except (ValueError, TypeError):
                pass

    if not issues:
        issues.append("No issues detected")

    return issues

def clean_data(df):
    # Handle columns with inconsistent units or formats
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: convert_to_numeric_if_needed(x))
                
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            elif df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)  # Generic fill for other types

    # Drop columns with only one unique value
    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(col, axis=1, inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    return df

def convert_to_numeric_if_needed(value):
    """Convert specific numeric values while leaving text-based data unchanged."""
    if isinstance(value, str):
        # Use patterns to differentiate between descriptive text and numeric values
        if re.search(r'\d+\s(BHK|Apartment|House|Villa)', value, re.IGNORECASE):
            return value  # Leave descriptive property names unchanged
        if re.search(r'Poss\.\sby\s\w+\s\'\d{2}', value, re.IGNORECASE):
            return value  # Leave date descriptions like 'Poss. by Oct '24' unchanged
        if re.search(r'\b(per\s)?(sqft|sqm|Lac|₹)\b', value, re.IGNORECASE):
            # Convert numeric parts, stripping out currency and unit indicators
            numeric_value = re.sub(r'[^\d.]', '', value)
            try:
                return float(numeric_value)
            except ValueError:
                return value  # If conversion fails, return the original value unchanged
    return value

# Example usage:
file_path = 'D:/AutoDash/laptopData.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Detect issues
issues = detect_issues(df)
print("Detected Issues:")
for issue in issues:
    print(f"- {issue}")

# Clean the data
cleaned_df = clean_data(df)
cleaned_df.to_csv('D:/AutoDash/w.csv', index=False)
print("\nData cleaned and saved to 'surat_cleaned.csv'.")