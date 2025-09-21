# Component 3: Exploratory Data Analysis (EDA)
# Project: Employee Performance Prediction
# Name: Nithin Aparadapu
# Date: 21-Sep-2025

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

# -----------------------------
# Step 2: Load Dataset
# -----------------------------
data = pd.read_csv('employee_data.csv')  # Replace with your actual CSV path

# Quick look at the data
print(data.head())
print(data.info())
print(data.describe())

# -----------------------------
# Step 3: Data Cleaning
# -----------------------------
# Check for missing values
print(data.isnull().sum())

# Fill numeric missing values with median
numeric_cols = data.select_dtypes(include=np.number).columns
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)

# Fill categorical missing values with mode
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Ensure correct data types
for col in categorical_cols:
    data[col] = data[col].astype('category')

# -----------------------------
# Step 4: Count Plots for Categorical Features
# -----------------------------

# 1. Department
plt.figure(figsize=(10,5))
sns.countplot(x='Department', data=data, palette='pastel')
plt.title('Number of Employees in Each Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('Department_Count.png', dpi=300)
plt.show()

# 2. Job Role
plt.figure(figsize=(12,5))
sns.countplot(x='JobRole', data=data, palette='pastel')
plt.title('Number of Employees in Each Job Role')
plt.xlabel('Job Role')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('JobRole_Count.png', dpi=300)
plt.show()

# 3. Performance Rating
plt.figure(figsize=(6,4))
sns.countplot(x='PerformanceRating', data=data, palette='pastel')
plt.title('Distribution of Employee Performance Ratings')
plt.xlabel('Performance Rating')
plt.ylabel('Count')
plt.savefig('Performance_Count.png', dpi=300)
plt.show()

# 4. Department vs Performance
plt.figure(figsize=(10,5))
sns.countplot(x='Department', hue='PerformanceRating', data=data, palette='Set2')
plt.title('Performance Ratings Across Departments')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Performance Rating')
plt.savefig('Dept_Performance.png', dpi=300)
plt.show()

# -----------------------------
# Step 5: Unique Values in Key Columns
# -----------------------------
print("Unique Departments:", data['Department'].unique())
print("Unique Job Roles:", data['JobRole'].unique())
print("Unique Performance Ratings:", data['PerformanceRating'].unique())

# -----------------------------
# Step 6: Save Cleaned Dataset (Optional)
# -----------------------------
data.to_csv('employee_data_cleaned.csv', index=False)