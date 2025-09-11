'''

HIT140 Assessment 2: Part I - Data Loading and Inspection

Group Name: SYDN 28
Group Members:
Krish Rajbhandari - S395754
Tasnim Zannat - S394294
Asma Zia - S395083
Suyog Kadariya - S393829

This code loads the two datasets into Python and checks them step by step. 
First it shows a small preview, then looks at info, missing values, and summary stats. 
The purpose is to understand the structure of the data before we do any cleaning or analysis.
    
'''


# Import libraries
import pandas as pd

print("Checking datasets...")

# Load the datasets
# dataset1.csv: Bat landings (individual events)
# dataset2.csv: 30-min observation periods
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")
print("Datasets loaded successfully.\n")

# Show first few rows for quick preview instead of whole dataset
print("Step 1: Preview of Dataset 1 (Bat Landings)")
print(dataset1.head(), "\n")

print("Step 1: Preview of Dataset 2 (Observation Periods)")
print(dataset2.head(), "\n")

# Show dataset information
print("Step 2: Dataset information")
print(dataset1.info(), "\n")
print(dataset2.info(), "\n")

# Check for missing values, if any
print("Step 3: Missing values check")
print("Dataset 1:\n", dataset1.isnull().sum(), "\n")
print("Dataset 2:\n", dataset2.isnull().sum(), "\n")

# Summary statistics
print("Step 4: Summary statistics")
print("Dataset 1:\n", dataset1.describe(include='all'), "\n")
print("Dataset 2:\n", dataset2.describe(include='all'), "\n")

print("Data inspection complete.")