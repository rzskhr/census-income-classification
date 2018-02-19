
# coding: utf-8

# # Predicting the Income bracket of a person Census Data

# In[1]:


# Importing dependencies
import pandas as pd
import numpy as np
from collections import defaultdict

# for visualizations
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

# For Hypothesis testing
import statsmodels.formula.api as smf

# For principal components analysis
from sklearn.decomposition import PCA


# ### Loading the Dataset

# In[2]:


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


# ---

# ## Problem Statement

# >From the various features in the census data set our aim is to build a predictive model to determine whether the income level for the people in United States exceeds the bracket of $50,000.

# ## Hypothesis Statement

# From our problem statement is clear that it is a binary classification problem.
# 
# Let us generate some hypotheses which will help us in building the models more efficiently. We need to figure out some hypotheses which might influence our final outcome, hence we need to answer a simple question.
# 
# **Is There a Relationship Between the Response and Predictors?**
# 
# To test this we use the test between the Null Hypothesis $H_0$ versus the Alternate Hypothesis $H_a$.
# * $H_0$ : There is no relationship between the response Income and the predictors.
#     
# * $H_a$ : There is some realtionship between the response and the predictors.
# 
# After building the model and fitting the data into our model, if the accuracy of the model beats the baseline and is statistically significant we will reject our Null Hypothesis $H_0$.
# 

# ## Baseline

# In order to evaluate our model we should define some baseline. Let us generate some statistics about our response variable so that we can set our baseline.

# In[3]:


# Getting the count
incomeCount = censusDf['targetIncome'].value_counts()
print(incomeCount)

# Getting the proportion of data having -50000 as response
print(float(incomeCount[0]/len(censusDf['targetIncome']))*100,
     "% people have income below $50000.")


# Most of the values are 0 in the responce variable, Income. Which means that the dataset is heavily skewed towards having income less than \$50,000. Which means that if we predict only below \$50,000, still our model accuracy would be **93.79%**.

# ---

# ## Data Wrangling

# ### 1. Missing Value Imputation

# In[4]:


censusDf.isnull().sum().sort_values(ascending=False).head()


# * We can observe from the above statistics that, there are no missing values in numerical columns of the dataset. 
# * There is only one column in which there are 874 missing values, which is 'HispanicOrigin'.
# * From the first five lines of dataframe displayed above we saw that there are some garbage/missing values in the dataframe labelled as '?', lets try to track them.

# In[5]:


# There are lot of '?' appearing in the dataset lets track them
for i in censusDf.columns:
    if '?' in list(censusDf[i]):
        print(censusDf.loc[censusDf[i].isin(['?'])][i].value_counts())


# The above missing values does not makes much sense if we substitute them, as they are nominal values. Let us label all the above missing values as 'Unavailable'. Also there are four columns in which there almost 50% of the values which are '?', it is better to drop those columns, as high proportion of missing values can be misleading.

# In[6]:


# Dropping the columns with missing values more than 50% and storing in a new dataframe
censusDf_cleaned = censusDf.drop(['MigrationCode_MSA', 'MigrationCode_REG', 
                                  'MigrationCode_WITHIN_REG', 
                                  'MigrationPrevResInSunbelt'], axis=1)

# Replacing the '?' with the label 'Unavailable'
censusDf_cleaned = censusDf_cleaned.replace('?', 'Unavailable')


# In[7]:


# Check if the values are replaced
for i in censusDf_cleaned.columns:
    if 'Unavailable' in list(censusDf_cleaned[i]):
        print(censusDf_cleaned.loc[censusDf_cleaned[i].isin(['Unavailable'])][i].value_counts())


# * As we saw earlier, for the caolumn 'HispanicOrigin' we have few (874) missing values; lets see how the values are distributed in the column, so that we can impute the missing values.

# In[8]:


censusDf_cleaned['HispanicOrigin'].value_counts().sort_values(ascending=False)


# Creating a new column for the missing values for HispanicOrigin.

# In[9]:


# store the missing value in a variable
missing_val = censusDf_cleaned[censusDf_cleaned.isnull()]['HispanicOrigin'].iloc[1]
# impute the missing values
censusDf_cleaned['HispanicOrigin'] = censusDf_cleaned['HispanicOrigin'].replace(
    missing_val, 'None')


# In[10]:


# Check if the values are replaced
censusDf_cleaned['HispanicOrigin'].value_counts()


# ##### **Check for missing values one last time.**

# In[11]:


# Check for missing values
censusDf_cleaned.isnull().sum().sort_values(ascending=False).head()


# > Now there are no missing values in the dataset.

# ### 2. Feature Engineering

# In[12]:


# Categorizing the columns

# Replacing the 'targetIncome' values with dummy variables
# - 50000. as the baseline. 0 for - 50000. and 1 for 50000+.
censusDf_cleaned['targetIncome'] = pd.get_dummies(
    censusDf_cleaned.targetIncome).iloc[:,1:]

# Features and Outcome
X = censusDf_cleaned.drop('targetIncome',1)
y = censusDf_cleaned.targetIncome
print("X (predictors) is ",X.shape[0],"rows,", X.shape[1],"columns, and..."      "\ny (response) is ",y.shape[0],"rows.")


# ##### **Let us check the categorical variables in for each feature, and decide which one to  use in our model.**

# In[13]:


# Print out number of unique categorical values in each column
print("NUMBER OF UNIQUE VALUES IN EACH FEATURE:\n")
for col_name in X.columns:
    if X[col_name].dtype == 'object':
        unique_val = len(X[col_name].unique())
        print("'{col_name}' has --> {unique_val}        ".format(col_name=col_name, unique_val=unique_val))


# ##### It looks like the columns 'BirthCountryFather', 'BirthCountryMother' and 'BirthCountrySelf' have same number of unique values. Let us keep only one column, and drop the other two.

# In[14]:


# Dropping the columns
X = X.drop(['BirthCountryFather', 'BirthCountryMother'], axis=1)
# keeping 'BirthCountrySelf' and renaming
X.rename(columns={'BirthCountrySelf': 'BirthCountry'}, inplace=True)


# In[15]:


# Although, 'BirthCountry' has a lot of unique categories, ...
# ...most categories only have a few observations if compared to max (United-States)
X['BirthCountry'].value_counts().sort_values(ascending=False).head(10)


# In[16]:


# In this case, bucket low frequecy categories as "Other"
X['BirthCountry'] = ['United-States' if x == 'United-States' 
                       else 'Other-Countries' for x in X['BirthCountry']]
# check the values
X['BirthCountry'].value_counts().sort_values(ascending=False)


# ##### The column 'HouseholdStatus' has 38 unique values; only few of the categories have significant number of observations.

# In[17]:


# Check the value counts
X['HouseholdStatus'].value_counts().sort_values(ascending=False).head(10)


# It is better to categorize the values as other, which does not have significant count.

# In[18]:


# Bucket the low frequency category as other
X['HouseholdStatus'] = ['Householder' if x == 'Householder'
                        else 'Children' if x == 'Child <18 never marr not in subfamily'
                        else 'Spouse' if x == 'Spouse of householder'
                        else 'Nonfamily' if x == 'Nonfamily householder'
                        else 'Child_18_plus' if x == 'Child 18+ never marr Not in a subfamily'
                        else 'Secondary_indv' if x == 'Secondary individual'
                       else 'Other_Householders' for x in X['HouseholdStatus']]
# check the values
X['HouseholdStatus'].value_counts().sort_values(ascending=False)


# ##### Lets check the 'PrevState' column, there are 51, unique values for the feature, lets see what are they.

# In[19]:


# Check the value counts
X['PrevState'].value_counts().sort_values(ascending=False).head(10)


# With approximately 200,000 rows in our dataset, there are almost 184,000 values for the 'PrevState' column, that say 'Not in universe', which is almost 96% of the entire row, since the survey has been conducted in the United States of America, all of them must belong to a state, hence the value stating "Not in universe" are the missing values. Having this much small information about the sate doesn't seem to be helpful, it is better that we drop this feature from our predictors variables list.

# In[20]:


# Dropping the 'PrevState' column
X = X.drop(['PrevState'], axis=1)


# #### Creating Dummies

# **Coverting categorical variable in to _Dummy Variables_.** If we want to include a categorical feature in our machine learning model, one common solution is to create dummy variables. We drop the original feature from the dataset and add a dummied version of the feature to the dataset, which is easier for the model to interpret.

# In[21]:


# Creating a list of categorical features to create a dummy variable of
# columns names in asscending order, according to number of diff unique values
features_to_dummy = ['Sex', 'BirthCountry', 'Year', 'EducationalInst', 
                     'MemLabourUnion', 'HouseOneYearAgo', 'OwnBusiness', 'VeteranQA',
                     'VeteranBenefits', 'Race', 'Parent', 'Citizenship', 
                     'UnemploymentReason', 'FEDERALTAX', 'TaxFilerStat', 
                     'MaritalStatus', 'HouseholdStatus', 'NumOfPersonForEmployer', 
                     'EmploymentStatus', 'HouseholdSummary', 'ClassOfWorker', 
                     'HispanicOrigin', 'OccupationCode', 'Education', 
                     'IndustryCode', 'Occupation', 'Industry','WeeksWorked']


# Define a function to create dummy variables of the dataframe from the list of columns.

# In[22]:


# Function to create the dummy categorical variables used for modeling
def create_dummies(df, col_name_list):
    """
    This function takes the dataframe and features list as input, 
    and returns the modified dataframe with dummy variables of the 
    features in the list col_name_list.
    
    :param df: target dataframe 
    :param col_name_list: list of the column names from the dataset
    :return: modifies the dataframe df inplace and returns dummied dataframe
             of features in col_name_list
    """
    for x in col_name_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


# Calling the function create_dummies to convert our features in to dummy variables.

# In[23]:


# Before dummies
print("Dataframe X has", X.shape[1],"columns",X.shape[0],"and rows.")

# Call the function create_dummies on X and replace the features with dummies
print("Creating dummies ...")
X = create_dummies(X, features_to_dummy)

# Printing the dimensions of the modified feature set
print("*** Now our dataframe has", X.shape[1],"columns",X.shape[0],"and rows. ***")

# display first five rows of all the features
with pd.option_context('display.max_columns', None):
    display(X.head())


# #### Principal Components Analysis

# Principal component analysis (PCA) transforms the dataset of many features into few Principal Components that "summarize" the variance underying in the data. It is the  most common way of dimensionality reduction, and it works well where the features are highly corelated. The drawback of using PCA is that it makes it difficult to interpret the data.

# In[24]:


# We will use PCA from sklearn.decomposition to find the principal components
pca = PCA(n_components=10) # 10 principal components
X_pca = pd.DataFrame(pca.fit_transform(X))


# In[25]:


# Displaying the first few rows of 10 pcs
X_pca.head()


# _In this case we will not proceed with the principal components. Because it is not recommended to perform PCA on categorical data._

# ***

# ### Model Building

# In[26]:


# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


# In[27]:


class ModelMetrics(object):

    # Random permutation cross-validator with 80-20 train test split
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state=1)
    # dictionary to store scores
    model_scores = dict()
    # default scoring metrics
    default_metric = ['accuracy','precision', 'recall']
    
    def __init__(self, model_name, model_obj, features, response):
        self.model_name = model_name
        self.model_obj = model_obj
        ModelMetrics.model_scores[model_name] = []
        self.cv = ModelMetrics.cv
        self.features = features
        self.response = response
        self.model_scores = ModelMetrics.model_scores
        
    def model_scoring(self, scoring_metric=default_metric):
        for metric in scoring_metric:
            n_fold_score = cross_val_score(self.model_obj,self.features,
                                                           self.response,
                                                           cv=self.cv,
                                                           scoring=metric)
            self.model_scores[self.model_name].append({metric:n_fold_score})
        model_scores = self.model_scores
        return model_scores


# In[28]:


def create_metric_df(metric_dict):
    """takes input as dict obtained from ModelMetrics
    class and maps the scores to a pandas dataframe
    :params: 
    metric_dict : dictionary of scores 
    :returns:
    score_df : pandas dataframe mapping the metrics and model
    """
    
    # use for loop to store the values
    model_name, acc_nfolds, pre_nfolds, rec_nfolds = (list() for i in range(4)) 
    for model, metrics in metric_dict.items():
        model_name.append(model)
        acc_nfolds.append(list(metrics[0].values())[0])
        pre_nfolds.append(list(metrics[1].values())[0]) 
        rec_nfolds.append(list(metrics[2].values())[0])
        
    metric_col_names = ['accuracy', 'precision', 'recall',
                'accuracy_nfolds', 'precision_nfolds', 'recall_nfolds']
    score_df = pd.DataFrame(columns = metric_col_names, index=model_name)

    # assign values
    score_df['accuracy_nfolds'] = acc_nfolds
    score_df['precision_nfolds'] = pre_nfolds
    score_df['recall_nfolds'] = rec_nfolds
    score_df['accuracy'] = score_df['accuracy_nfolds'].apply(np.mean)
    score_df['precision'] = score_df['precision_nfolds'].apply(np.mean)
    score_df['recall'] = score_df['recall_nfolds'].apply(np.mean)
    
    return score_df


# #### Generate plots

# In[29]:


def plot_metric_each_fold(metric, title):
    metric = metric+"_nfolds"
    legend_list = list()
    for i,model in enumerate(list(final_results.index)):
        x_val = list("Fold-"+str(i) 
                     for i in range(len(final_results[metric][0])))
        y_val = final_results[metric][i]
        
        # fig size
        plt.rcParams["figure.figsize"] = (16, 9)
        plt.rcParams.update({'font.size': 15})
        plt.plot(x_val,y_val,
                marker=".", markeredgewidth=1,linestyle=":", linewidth=3.5)
        for a,b in zip(x_val,y_val): 
            plt.text(a, b, str(b))
        legend_list.append(model)
        plt.legend(legend_list)
        plt.title(title)

    # add a horiontal line for baseline
    if(metric == "accuracy_nfolds"):
        baseline = float(y.value_counts()[0]/len(y))
        plt.axhline(y=baseline, color='r', linestyle='-')


# #### Logistic Regression Model

# In[30]:


# Classifier implementing Logistic Regression
clf_log_reg = LogisticRegression()
# Creating object for metrics
log_reg_metrics_obj = ModelMetrics("Logistic Regression", clf_log_reg, X, y)
# get the metrics
log_reg_metrics = log_reg_metrics_obj.model_scoring()


# #### Decision Tree

# In[31]:


# Classifier implementing the Decision Tree
clf_d_tree = DecisionTreeClassifier()
# Creating object for metrics
clf_d_tree_metrics_obj = ModelMetrics("Decision Tree", clf_d_tree, X, y)
# get the metrics
clf_d_tree_metrics = clf_d_tree_metrics_obj.model_scoring()


# #### Random Forest Classifier

# In[32]:


# Classifier implementing  Random Forest Classifier
clf_RF = RandomForestClassifier()
# Creating object for metrics
clf_RF_metrics_obj = ModelMetrics("Random Forest Classifier", clf_RF, X, y)
# get the metrics
clf_RF_metrics = clf_RF_metrics_obj.model_scoring()


# #### KNN

# In[34]:


# Classifier implementing the k-nearest neighbors
clf_knn = KNeighborsClassifier()
# Creating object for metrics
clf_knn_metrics_obj = ModelMetrics("k-nearest neighbors", clf_knn, X, y)
# get the metrics
clf_knn_metrics = clf_knn_metrics_obj.model_scoring()


# #### Store the results in a DataFrame object

# In[36]:


# sotore the results in a data frame
final_results = create_metric_df(ModelMetrics.model_scores)
# print the average score for all folds
final_results.iloc[:,0:3]


# In[37]:


# scores for n folds
pd.set_option('display.max_colwidth', -1)
final_results.iloc[:,3:]


# ### Generate Plots

# Plot the average accuracy for all the folds for all models.

# In[38]:


# create a scatter plot of accuracy for all the models
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams.update({'font.size': 15})
plt.scatter(final_results.index, final_results["accuracy"], 
            100,alpha=1, marker='*')
plt.axhline(y=float(y.value_counts()[0]/len(y)), 
            color='r', linestyle='-') # horizintal basline
plt.legend()


# ### The confusion matrix.
# Class 0 = Income is less than 50k.<br/>
# Class 1 = Income is More than 50k.<br/>
# ![title](../data/conf-mat.png)

# > **Plotting all the scoring metrics for all the classifiers for each fold.**

# ### Accuracy
# 
# Accuracy is the metric for the % of correct prediction. It is defined as the given formula below. It is simply a ratio of correctly predicted observation to the total observations.<br/>
# 
# $Accuracy = \dfrac{Total Number of correct classifications}{Total Number of Cases} = \dfrac{TP + TN}{TP + TN + FP + FN}$

# In[39]:


metric = "accuracy"
title = "Compare Accuracy of all the models at each fold"
plot_metric_each_fold(metric, title)


# Out of all the models we used for classification all the accuracy score beats the baseline we set earlier, except of Decision Tree, whose accuracy is **93.3662%**, which is slightly less than what we expected from our model to perform. Apart from decision tree all the models seems to do a good job of classyfying the outcomes, which is great.

# ### Precision
# 
# Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. Which means out of all Class 0, how many we predicted as Class 0. A high precision relates to the low false positive rate.<br/><br/>
# $Precision= \dfrac{Total Number of Actual Positive classifications}{Total Number of Predicted Positive Classes} = \dfrac{TP}{TP + FP}$
# 

# In[40]:


metric = "precision"
title = "Compare Precision of all the models at each fold"
plot_metric_each_fold(metric, title)


# If our precision score is **0.61454525** (which is our average precision score for all the models) this says that if our classifier predicts someone to have an income below 50K is right about **61.45%** of the time. This is not bad because the number of population with income more than 50k are fewer than that of less than 50K.

# ### Recall
# 
# Recall(Sensitivity) is the ratio of correctly predicted positive observations to the all observations in actual class.<br/><br/>
# 
# $Recall= \dfrac{Total Number of Actual Positive classifications}{Total Number of actual Positive Classes} = \dfrac{TP}{TP + FN}$
# 

# In[41]:


metric = "recall"
title = "Compare Recall of all the models at each fold"
plot_metric_each_fold(metric, title)


# How many positives do I retrieve and how many do I miss. That is recall. So if the recall score is **0.36515075** (wchich is our average recall score for all the models), it means that the classifier predicts **36.51%** of the actual Class 0 correctly and looses **63.49%** of them.

# ---
