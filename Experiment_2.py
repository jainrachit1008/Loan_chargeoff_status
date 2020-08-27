from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import statsmodels.api as sm
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# Data Extraction
train = pd.read_csv('lc_loan.csv')
test = pd.read_csv('lc_2016_2017.csv')

# Cleaning the data
train = train.replace(0.0,np.nan)
train = train.dropna(axis=1,thresh = 0.8*len(train))
to_be_del = ['emp_title','emp_length','policy_code','title','zip_code','addr_state','url','issue_d','pymnt_plan','last_pymnt_d','last_pymnt_amnt','last_credit_pull_d']
train = train.drop(columns = to_be_del)
train.head()

# Imputing the missing values
cols = ['funded_amnt_inv','earliest_cr_line','open_acc','revol_bal','revol_util','total_acc','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','tot_cur_bal','total_rev_hi_lim']
train.update(train[cols].fillna(1))
train['dti'].fillna(train['dti'].mean(), inplace=True)
train.isnull().sum()

# Check some data stats
train.loan_status.value_counts()
train.describe()

twotables = pd.crosstab(train["int_rate"], train["loan_status"], margins=True)
twotables

twotables1 = pd.crosstab(train["home_ownership"], train["loan_status"], margins=True)
twotables1.plot(kind = 'bar')
plt.title('Home Ownership for each loan status')
plt.xlabel('Home Ownership status')
plt.ylabel('Frequency of loan status')

twotables2 = pd.crosstab(train["term"], train["loan_status"], margins=True)
twotables2.plot(kind = 'bar')
plt.title('Terms for each loan status')
plt.xlabel('Month')
plt.ylabel('Frequency of loan status')

Ann_Income_boxplt = train.boxplot(column="annual_inc",by="loan_status")
Ann_Income_boxplt.figure.savefig('Ann_Income_boxplt.png')
Ann_Income_boxplt

train.hist(column="annual_inc",by="loan_status",bins=30)

# Relationship of Revolving Balance with loan_status
train.boxplot(column="revol_bal",by="loan_status")

Log - Transformation of numerical predictors

train['annual_inc_log']= np.log(train['annual_inc'])
print (train['annual_inc_log'].hist(bins = 50))

train['revol_bal_log']= np.log(train['revol_bal'])
# plt.hist(train[np.isfinite(train['revol_bal'])].values)
print (train['revol_bal_log'].hist(bins = 100))

train['int_rate_log']= np.log(train['int_rate'])
print (train['int_rate_log'].hist(bins = 50))

train['revol_util_log']= np.log(train['revol_util'])
print (train['revol_util_log'].hist(bins = 50))

train['dti_log']= np.log(train['dti'])
print (train['dti_log'].hist(bins = 50))

Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables. To represent binary / categorical outcome, we use dummy variables. You can also think of logistic regression as a special case of linear regression when the outcome variable is categorical, where we are using log of odds as the dependent variable.
In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function, read more about Logistic Regression .
LogisticRegression() function is part of linear_model module of sklearn and is used to create logistic regression

## Model Building
from sklearn.preprocessing import LabelEncoder
#"Scikit Learn" library has a module called "LabelEncoder" which helps 
# to label character labels into numbers so first import module "LabelEncoder".
number = LabelEncoder()
# Transform variable to dummy variables
train['term_new'] = number.fit_transform(train['term'].astype(str))
train['home_ownership_new'] = number.fit_transform(train['home_ownership'].astype(str))
train['grade_new'] = number.fit_transform(train['grade'].astype(str))
train['loan_status_new'] = number.fit_transform(train['loan_status'].astype(str))

# Import linear model of sklearn
import sklearn.linear_model
 
# Create object of Logistic Regression
model=sklearn.linear_model.LogisticRegression()

# Select the predictors
predictors =['term_new','home_ownership_new','grade_new','int_rate_log','revol_util_log','dti_log']

# Converting predictors and outcome to numpy array
x_train = train[predictors].values
y_train = train['loan_status_new'].values

# Model Building
model.fit(x_train, y_train)

print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(model.score(x_train, y_train)))

Transforming the predictors for Test Data

# Cleaning the data
test = test.replace(0.0,np.nan)
test = test.dropna(axis=1,thresh = 0.8*len(test))
to_be_del = ['open_rv_24m','mths_since_rcnt_il','zip_code','emp_title','emp_length','title','last_pymnt_d','last_pymnt_amnt','last_credit_pull_d']
test = test.drop(columns = to_be_del)

# Imputing the missing values
cols = ['all_util','max_bal_bc','total_bal_il','il_util','annual_inc','funded_amnt_inv','earliest_cr_line','open_acc','revol_bal','revol_util','total_acc','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','tot_cur_bal','total_rev_hi_lim']
test.update(test[cols].fillna(1))
test['dti'].fillna(test['dti'].mean(), inplace=True)
test.isnull().sum()

test['int_rate_log']= np.log(test['int_rate'])
test['revol_util_log']= np.log(test['revol_util'])
test['dti_log']= np.log(test['dti'])
test['term_new'] = number.fit_transform(test['term'].astype(str))
test['home_ownership_new'] = number.fit_transform(test['home_ownership'].astype(str))
test['grade_new'] = number.fit_transform(test['grade'].astype(str))
y_test = number.fit_transform(test['loan_status'].astype(str))

# Converting predictors and outcome to numpy array
predictors =['term_new','home_ownership_new','grade_new','int_rate_log','revol_util_log','dti_log']
x_test = test[predictors].values
np.isnan(x_test)
np.where(np.isnan(x_test))
x_test = np.nan_to_num(x_test)

#Predict Output
y_pred= model.predict(x_test)

#Reverse encoding for predicted outcome
y_pred = number.inverse_transform(y_pred)

#Store it to test dataset
test['Loan_Status_new']=y_pred

#Output file to make submission
test.to_csv("TestFile_Submission.csv",columns=['id','Loan_Status_new'])

unique, counts = np.unique(y_pred, return_counts=True)
print (unique, counts)

y_test = number.inverse_transform(y_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual loan status')
plt.xlabel('Predicted loan status')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,pos_label='positive',
                                           average='micro'))
print("Recall:",metrics.recall_score(y_test, y_pred,pos_label='positive',
                                           average='micro'))

test['loan_status'].value_counts()






