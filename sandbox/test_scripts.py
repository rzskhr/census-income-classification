
# Importing dependencies
import pandas as pd
import numpy as np


# For Hypothesis testing
import statsmodels.formula.api as smf

# Loading the dataset into pandas dataframe
# path = "../data/census-income.data.gz"
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

# Some statistics about the data
censusDf.describe()

