# Importing dependencies
import pandas as pd
import numpy as np

# for visualizations
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'

# For Hypothesis testing
import statsmodels.formula.api as smf


# Loading the dataset into pandas dataframe
path = "../data/census-income.data.gz"
# set the column names
censusColnames = ['Age', 'ClassOfWorker', 'Industry', 'Occupation', 'Education',
                  'WagePerHr', 'EducationalInst', 'MaritalStatus', 'IndustryCode',
                  'OccupationCode', 'Race', 'HispanicOrigin', 'Sex', 'MemLabourUnion',
                  'UnemploymentReason', 'EmploymentStatus', 'CapitalGain', 'CapitalLoss',
                  'Dividends', 'FEDERALTAX', 'TaxFilerStat', 'PrevState',
                  'HouseholdStatus', 'HouseholdSummary', 'INSTANCEWEIGHT',
                  'MigrationCode_MSA', 'MigrationCode_REG',
                  'MigrationCode_WITHIN_REG', 'HouseOneYearAgo',
                  'MigrationPrevResInSunbelt', 'NumOfPersonForEmployer', 'Parent',
                  'BirthCountryFather', 'BirthCountryMother', 'BirthCountrySelf',
                  'Citizenship', 'OwnBusiness', 'VeteranQA', 'VeteranBenefits',
                  'WeeksWorked', 'Year', 'targetIncome']
censusDf = pd.read_csv(path, sep=r',', skipinitialspace=True,
                       names = censusColnames, header='infer')

# Printing the dimensions of the dataset
print(censusDf.shape[0],"rows,", censusDf.shape[1],"columns")

# Displaying first five elements of all columns
with pd.option_context('display.max_columns', None):
    display(censusDf.head())


# Baseline

# Getting the count
incomeCount = censusDf['targetIncome'].value_counts()
print(incomeCount)

# Getting the proportion of data having -50000 as response
print(float(incomeCount[0]/len(censusDf['targetIncome']))*100,
     "% people have income below $50000.")

# Most of the values are 0 in the responce variable, Income. Which means that the dataset is heavily skewed towards
# having income less than \$50,000. Which means that if we predict only below \$50,000, still our model accuracy would
# be 93.79%.


# Data Wrangling
censusDf.isnull().sum().sort_values(ascending=False).head()

# There are lot of '?' appearing in the dataset lets track them
for i in censusDf.columns:
    if '?' in list(censusDf[i]):
        print(censusDf.loc[censusDf[i].isin(['?'])][i].value_counts())

# Dropping the columns with missing values more than 50% and storing in a new dataframe
censusDf_cleaned = censusDf.drop(['MigrationCode_MSA', 'MigrationCode_REG',
                                  'MigrationCode_WITHIN_REG',
                                  'MigrationPrevResInSunbelt'], axis=1)

# Replacing the '?' with the label 'Unavailable'
censusDf_cleaned = censusDf_cleaned.replace('?', 'Unavailable')

# Check if the values are replaced
for i in censusDf_cleaned.columns:
    if 'Unavailable' in list(censusDf_cleaned[i]):
        print(censusDf_cleaned.loc[censusDf_cleaned[i].isin(['Unavailable'])][i].value_counts())


censusDf_cleaned['HispanicOrigin'].value_counts().sort_values(ascending=False)

# store the missing value in a variable
missing_val = censusDf_cleaned[censusDf_cleaned.isnull()]['HispanicOrigin'].iloc[1]
# replace the missing values
censusDf_cleaned['HispanicOrigin'] = censusDf_cleaned['HispanicOrigin'].replace(
    missing_val, 'All other')


# Check for missing values
censusDf_cleaned.isnull().sum().sort_values(ascending=False).head()




# Feature Engineering

# Categorizing the columns

# Continuous Features
ordinalFeatures = ['Age', 'WagePerHr', 'CapitalGain', 'CapitalLoss','Dividends',
     'INSTANCEWEIGHT', 'NumOfPersonForEmployer', 'WeeksWorked']

# Nominal Features
nominalFeatures = ['ClassOfWorker', 'Industry', 'Occupation', 'Education',
                  'EducationalInst', 'MaritalStatus', 'IndustryCode', 'OccupationCode',
                  'Race', 'HispanicOrigin', 'Sex', 'MemLabourUnion',
                  'UnemploymentReason', 'EmploymentStatus','FEDERALTAX',
                   'TaxFilerStat', 'PrevState', 'HouseholdStatus',
                  'HouseholdSummary', 'LiveInHouse',
                  'Parent', 'BirthCountryFather', 'BirthCountryMother',
                  'BirthCountrySelf', 'Citizenship', 'OwnBusiness', 'VeteranQA', 'VeteranBenefits',
                  'Year', 'targetIncome']

# Replacing the 'targetIncome' values with dummy variables
# - 50000. as the baseline. 0 for - 50000. and 1 for 50000+.
censusDf_cleaned['targetIncome'] = pd.get_dummies(censusDf_cleaned.targetIncome).iloc[:,1:]

# Check the features
print(len(censusDf_cleaned.columns) == len(ordinalFeatures) + len(nominalFeatures))

# Features and Outcome
X = censusDf_cleaned.drop('targetIncome',1)
Y = censusDf_cleaned.targetIncome
print("X (predictors) is ",X.shape[0],"rows,", X.shape[1],"columns, and..."\
      "\nY (response) is ",Y.shape[0],"rows.")


# Print out number of unique categorical values in each column
print("NUMBER OF UNIQUE VALUES IN EACH FEATURE:\n")
for col_name in X.columns:
    if X[col_name].dtype == 'object':
        unique_val = len(X[col_name].unique())
        print("'{col_name}' has --> {unique_val}\
        ".format(col_name=col_name, unique_val=unique_val))


# Dropping the columns
X = X.drop(['BirthCountryFather', 'BirthCountryMother'], axis=1)
# keeping 'BirthCountrySelf' and renaming
X.rename(columns={'BirthCountrySelf': 'BirthCountry'}, inplace=True)


# Although, 'BirthCountry' has a lot of unique categories, ...
# ...most categories only have a few observations if compared to max (United-States)
X['BirthCountry'].value_counts().sort_values(ascending=False).head(10)


# In this case, bucket low frequecy categories as "Other"
X['BirthCountry'] = ['United-States' if x == 'United-States'
                       else 'Other-Countries' for x in X['BirthCountry']]
# check the values
X['BirthCountry'].value_counts().sort_values(ascending=False)







