# PD model
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import scipy.stats as stat
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

# import data,
# to preserve the index we create a index column so data can be combined based on index if any shuffling done

loan_data_inputs_train = pd.read_csv(
    "C:\\Users\\mayan\\Downloads\\loan_data_inputs_train.csv", index_col=0)
loan_data_inputs_test = pd.read_csv(
    "C:\\Users\\mayan\\Downloads\\loan_data_inputs_test.csv", index_col=0)
loan_data_targets_train = pd.read_csv(
    "C:\\Users\\mayan\\Downloads\\loan_data_targets_train.csv", index_col=0)
loan_data_targets_test = pd.read_csv(
    "C:\\Users\\mayan\\Downloads\\loan_data_targets_test.csv", index_col=0)


# now we will create the dummy variables for the regression, we need to create k-1 dummy variables , to avoid the dummy variable trap, basically the reference category (the one with the lowest woe) will be removed and other dumies will be be kept when value of all dummies is 0 in a category it referes to the reference dummy variable

print(list(loan_data_inputs_train.columns))


# Here we select a limited set of input variables in a new dataframe.
inputs_train_with_ref_cat = loan_data_inputs_train.loc[:, ['grade:A',
                                                           'grade:B',
                                                           'grade:C',
                                                           'grade:D',
                                                           'grade:E',
                                                           'grade:F',
                                                           'grade:G',
                                                           'home_ownership:RENT_OTHER_NONE_ANY',
                                                           'home_ownership:OWN',
                                                           'home_ownership:MORTGAGE',
                                                           'addr_state:ND_NE_IA_NV_FL_HI_AL',
                                                           'addr_state:NM_VA',
                                                           'addr_state:NY',
                                                           'addr_state:OK_TN_MO_LA_MD_NC',
                                                           'addr_state:CA',
                                                           'addr_state:UT_KY_AZ_NJ',
                                                           'addr_state:AR_MI_PA_OH_MN',
                                                           'addr_state:RI_MA_DE_SD_IN',
                                                           'addr_state:GA_WA_OR',
                                                           'addr_state:WI_MT',
                                                           'addr_state:TX',
                                                           'addr_state:IL_CT',
                                                           'addr_state:KS_SC_CO_VT_AK_MS',
                                                           'addr_state:WV_NH_WY_DC_ME_ID',
                                                           'verification_status:Not Verified',
                                                           'verification_status:Source Verified',
                                                           'verification_status:Verified',
                                                           'purpose:educ__sm_b__wedd__ren_en__mov__house',
                                                           'purpose:credit_card',
                                                           'purpose:debt_consolidation',
                                                           'purpose:oth__med__vacation',
                                                           'purpose:major_purch__car__home_impr',
                                                           'initial_list_status:f',
                                                           'initial_list_status:w',
                                                           'term_36',
                                                           'term_60',
                                                           'emp_length:0',
                                                           'emp_length:1',
                                                           'emp_length:2-4',
                                                           'emp_length:5-6',
                                                           'emp_length:7-9',
                                                           'emp_length:10',
                                                           'mths_since_issue_d:<38',
                                                           'mths_since_issue_d:<38-39',
                                                           'mths_since_issue_d:<40-41',
                                                           'mths_since_issue_d:<42-48',
                                                           'mths_since_issue_d:<49-52',
                                                           'mths_since_issue_d:<53-64',
                                                           'mths_since_issue_d:<65-84',
                                                           'mths_since_issue_d:>84',
                                                           'int_rate:<9.548',
                                                           'int_rate:9.548-12.025',
                                                           'int_rate:12.025-15.74',
                                                           'int_rate:15.74-20.281',
                                                           'int_rate:>20.281',
                                                           'mths_since_earliest_cr_line:<140',
                                                           'mths_since_earliest_cr_line:141-164',
                                                           'mths_since_earliest_cr_line:165-247',
                                                           'mths_since_earliest_cr_line:248-270',
                                                           'mths_since_earliest_cr_line:271-352',
                                                           'mths_since_earliest_cr_line:>352',
                                                           'delinq_2yrs:0',
                                                           'delinq_2yrs:1-3',
                                                           'delinq_2yrs:>=4',
                                                           'inq_last_6mths:0',
                                                           'inq_last_6mths:1-2',
                                                           'inq_last_6mths:3-6',
                                                           'inq_last_6mths:>6',
                                                           'open_acc:0',
                                                           'open_acc:1-3',
                                                           'open_acc:4-12',
                                                           'open_acc:13-17',
                                                           'open_acc:18-22',
                                                           'open_acc:23-25',
                                                           'open_acc:26-30',
                                                           'open_acc:>=31',
                                                           'pub_rec:0-2',
                                                           'pub_rec:3-4',
                                                           'pub_rec:>=5',
                                                           'total_acc:<=27',
                                                           'total_acc:28-51',
                                                           'total_acc:>=52',
                                                           'acc_now_delinq:0',
                                                           'acc_now_delinq:>=1',
                                                           'total_rev_hi_lim:<=5K',
                                                           'total_rev_hi_lim:5K-10K',
                                                           'total_rev_hi_lim:10K-20K',
                                                           'total_rev_hi_lim:20K-30K',
                                                           'total_rev_hi_lim:30K-40K',
                                                           'total_rev_hi_lim:40K-55K',
                                                           'total_rev_hi_lim:55K-95K',
                                                           'total_rev_hi_lim:>95K',
                                                           'annual_inc:<20K',
                                                           'annual_inc:20K-30K',
                                                           'annual_inc:30K-40K',
                                                           'annual_inc:40K-50K',
                                                           'annual_inc:50K-60K',
                                                           'annual_inc:60K-70K',
                                                           'annual_inc:70K-80K',
                                                           'annual_inc:80K-90K',
                                                           'annual_inc:90K-100K',
                                                           'annual_inc:100K-120K',
                                                           'annual_inc:120K-140K',
                                                           'annual_inc:>140K',
                                                           'dti:<=1.4',
                                                           'dti:1.4-3.5',
                                                           'dti:3.5-7.7',
                                                           'dti:7.7-10.5',
                                                           'dti:10.5-16.1',
                                                           'dti:16.1-20.3',
                                                           'dti:20.3-21.7',
                                                           'dti:21.7-22.4',
                                                           'dti:22.4-35',
                                                           'dti:>35',
                                                           'mths_since_last_delinq:Missing',
                                                           'mths_since_last_delinq:0-3',
                                                           'mths_since_last_delinq:4-30',
                                                           'mths_since_last_delinq:31-56',
                                                           'mths_since_last_delinq:>=57',
                                                           'mths_since_last_record:Missing',
                                                           'mths_since_last_record:0-2',
                                                           'mths_since_last_record:3-20',
                                                           'mths_since_last_record:21-31',
                                                           'mths_since_last_record:32-80',
                                                           'mths_since_last_record:81-86',
                                                           'mths_since_last_record:>86',
                                                           ]]

# Here we store the names of the reference category dummy variables in a list.
ref_categories = ['grade:G',
                  'home_ownership:RENT_OTHER_NONE_ANY',
                  'addr_state:ND_NE_IA_NV_FL_HI_AL',
                  'verification_status:Verified',
                  'purpose:educ__sm_b__wedd__ren_en__mov__house',
                  'initial_list_status:f',
                  'term_60',
                  'emp_length:0',
                  'mths_since_issue_d:>84',
                  'int_rate:>20.281',
                  'mths_since_earliest_cr_line:<140',
                  'delinq_2yrs:>=4',
                  'inq_last_6mths:>6',
                  'open_acc:0',
                  'pub_rec:0-2',
                  'total_acc:<=27',
                  'acc_now_delinq:0',
                  'total_rev_hi_lim:<=5K',
                  'annual_inc:<20K',
                  'dti:>35',
                  'mths_since_last_delinq:0-3',
                  'mths_since_last_record:0-2']


# From the dataframe with input variables, we drop the variables with variable names in the list with reference categories.
inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis=1)

inputs_train.head()

# Logistic regression model for PD


# instance of log reg class
reg = LogisticRegression()

pd.options.display.max_rows = None

# fitting the model
reg.fit(inputs_train, loan_data_targets_train)


# to present the results we will extract the variable names, coeffiencts and p values and put in summary table

reg.intercept_  # intercept b0
reg.coef_  # coefficients

feature_name = inputs_train.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
# we use the transpose fn to convert row result into a column as reg.coef_ will give result in a single row
summary_table['coefficients'] = np.transpose(reg.coef_)
# we are adding 1 to each index value so that we can add intercept at index 0
summary_table.index = summary_table.index + 1
# adding intercept at the top of summary table
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
# ensuring index 0, intercept is at top
summary_table = summary_table.sort_index()
summary_table

# adding p values to the table
# sklearn have some methods to cal p value but they are univariate(they take into account impact of each variable on the outcome, as if there werent any other feature but in a reg model impact of all variables is collective rather than independent) log reg doesnot have a method to cal multivariate p values, we will alter the fit method from the log reg itself, the below code for class has been taken from internet


class LogisticRegression_with_p_values:

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = linear_model.LogisticRegression(
            *args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)

        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)  # Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  # Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        # z-score for eaach model coefficient
        z_scores = self.model.coef_[0] / sigma_estimates
        # two tailed test for p-values
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


reg = LogisticRegression_with_p_values()
# We create an instance of an object from the newly created 'LogisticRegression_with_p_values()' class.

# debugging
# print("Shape of inputs_train:", inputs_train.shape)
# print(inputs_train.head())  # Print a sample of the data
# print(inputs_train.dtypes)
# print("Missing values in inputs_train:")
# print(inputs_train.isnull().sum())

# print("Shape of loan_data_targets_train:", loan_data_targets_train.shape)
# print("Sample values in loan_data_targets_train:", loan_data_targets_train[:5])
# print("Any missing values in loan_data_targets_train:", pd.isnull(loan_data_targets_train).any())


# found that inputs_train has boolean values and loan_data_targets_train is 2D array needs to be 1d array
inputs_train = inputs_train.astype(int)
loan_data_targets_train_1D = loan_data_targets_train.values.ravel()


reg.fit(inputs_train, loan_data_targets_train_1D)
# Estimates the coefficients of the object from the 'LogisticRegression' class
# with inputs (independent variables) contained in the first dataframe
# and targets (dependent variables) contained in the second dataframe.


# Same as above.
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


p_values = reg.p_values
# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.

# Add the intercept for completeness.
p_values = np.append(np.nan, np.array(p_values))
# We add the value 'NaN' in the beginning of the variable with p-values, because p values are for coeff not for intercept

summary_table['p_values'] = p_values
# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable..

summary_table

# now we select the significant variables, if one coeff of var in a category is significant(pvalue) we must keep all var in that category, if non is sig we remove all of them- this was done in excel, sig level 1%, 5%, 10%

# for the final model , we remove: delinq_2yrs, open_acc, pub_rec, total_rev_hi_lim, total_acc


# We do that by specifying another list of dummy variables as reference categories, and a list of variables to remove.
# Then, we are going to drop the two datasets from the original list of dummy variables.

# Variables- all variables including the refrence variable, excluding categories delinq_2yrs, open_acc, pub_rec, total_rev_hi_lim, total_acc
inputs_train_with_ref_cat = loan_data_inputs_train.loc[:, ['grade:A',
                                                           'grade:B',
                                                           'grade:C',
                                                           'grade:D',
                                                           'grade:E',
                                                           'grade:F',
                                                           'grade:G',
                                                           'home_ownership:RENT_OTHER_NONE_ANY',
                                                           'home_ownership:OWN',
                                                           'home_ownership:MORTGAGE',
                                                           'addr_state:ND_NE_IA_NV_FL_HI_AL',
                                                           'addr_state:NM_VA',
                                                           'addr_state:NY',
                                                           'addr_state:OK_TN_MO_LA_MD_NC',
                                                           'addr_state:CA',
                                                           'addr_state:UT_KY_AZ_NJ',
                                                           'addr_state:AR_MI_PA_OH_MN',
                                                           'addr_state:RI_MA_DE_SD_IN',
                                                           'addr_state:GA_WA_OR',
                                                           'addr_state:WI_MT',
                                                           'addr_state:TX',
                                                           'addr_state:IL_CT',
                                                           'addr_state:KS_SC_CO_VT_AK_MS',
                                                           'addr_state:WV_NH_WY_DC_ME_ID',
                                                           'verification_status:Not Verified',
                                                           'verification_status:Source Verified',
                                                           'verification_status:Verified',
                                                           'purpose:educ__sm_b__wedd__ren_en__mov__house',
                                                           'purpose:credit_card',
                                                           'purpose:debt_consolidation',
                                                           'purpose:oth__med__vacation',
                                                           'purpose:major_purch__car__home_impr',
                                                           'initial_list_status:f',
                                                           'initial_list_status:w',
                                                           'term_36',
                                                           'term_60',
                                                           'emp_length:0',
                                                           'emp_length:1',
                                                           'emp_length:2-4',
                                                           'emp_length:5-6',
                                                           'emp_length:7-9',
                                                           'emp_length:10',
                                                           'mths_since_issue_d:<38',
                                                           'mths_since_issue_d:<38-39',
                                                           'mths_since_issue_d:<40-41',
                                                           'mths_since_issue_d:<42-48',
                                                           'mths_since_issue_d:<49-52',
                                                           'mths_since_issue_d:<53-64',
                                                           'mths_since_issue_d:<65-84',
                                                           'mths_since_issue_d:>84',
                                                           'int_rate:<9.548',
                                                           'int_rate:9.548-12.025',
                                                           'int_rate:12.025-15.74',
                                                           'int_rate:15.74-20.281',
                                                           'int_rate:>20.281',
                                                           'mths_since_earliest_cr_line:<140',
                                                           'mths_since_earliest_cr_line:141-164',
                                                           'mths_since_earliest_cr_line:165-247',
                                                           'mths_since_earliest_cr_line:248-270',
                                                           'mths_since_earliest_cr_line:271-352',
                                                           'mths_since_earliest_cr_line:>352',
                                                           'inq_last_6mths:0',
                                                           'inq_last_6mths:1-2',
                                                           'inq_last_6mths:3-6',
                                                           'inq_last_6mths:>6',
                                                           'acc_now_delinq:0',
                                                           'acc_now_delinq:>=1',
                                                           'annual_inc:<20K',
                                                           'annual_inc:20K-30K',
                                                           'annual_inc:30K-40K',
                                                           'annual_inc:40K-50K',
                                                           'annual_inc:50K-60K',
                                                           'annual_inc:60K-70K',
                                                           'annual_inc:70K-80K',
                                                           'annual_inc:80K-90K',
                                                           'annual_inc:90K-100K',
                                                           'annual_inc:100K-120K',
                                                           'annual_inc:120K-140K',
                                                           'annual_inc:>140K',
                                                           'dti:<=1.4',
                                                           'dti:1.4-3.5',
                                                           'dti:3.5-7.7',
                                                           'dti:7.7-10.5',
                                                           'dti:10.5-16.1',
                                                           'dti:16.1-20.3',
                                                           'dti:20.3-21.7',
                                                           'dti:21.7-22.4',
                                                           'dti:22.4-35',
                                                           'dti:>35',
                                                           'mths_since_last_delinq:Missing',
                                                           'mths_since_last_delinq:0-3',
                                                           'mths_since_last_delinq:4-30',
                                                           'mths_since_last_delinq:31-56',
                                                           'mths_since_last_delinq:>=57',
                                                           'mths_since_last_record:Missing',
                                                           'mths_since_last_record:0-2',
                                                           'mths_since_last_record:3-20',
                                                           'mths_since_last_record:21-31',
                                                           'mths_since_last_record:32-80',
                                                           'mths_since_last_record:81-86',
                                                           'mths_since_last_record:>86',
                                                           ]]

# all reference var excluding reference variables for delinq_2yrs, open_acc, pub_rec, total_rev_hi_lim, total_acc

ref_categories = ['grade:G',
                  'home_ownership:RENT_OTHER_NONE_ANY',
                  'addr_state:ND_NE_IA_NV_FL_HI_AL',
                  'verification_status:Verified',
                  'purpose:educ__sm_b__wedd__ren_en__mov__house',
                  'initial_list_status:f',
                  'term_60',
                  'emp_length:0',
                  'mths_since_issue_d:>84',
                  'int_rate:>20.281',
                  'mths_since_earliest_cr_line:<140',
                  'inq_last_6mths:>6',
                  'acc_now_delinq:0',
                  'annual_inc:<20K',
                  'dti:>35',
                  'mths_since_last_delinq:0-3',
                  'mths_since_last_record:0-2']

inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis=1)
inputs_train.head()

# found that inputs_train has boolean values and loan_data_targets_train is 2D array needs to be 1d array
inputs_train.dtypes
inputs_train = inputs_train.astype(int)
loan_data_targets_train_1D = loan_data_targets_train.values.ravel()

# new model with significant variables
reg2 = LogisticRegression_with_p_values()
reg2.fit(inputs_train, loan_data_targets_train_1D)

feature_name = inputs_train.columns.values


# summary table same as before
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg2.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg2.intercept_[0]]
summary_table = summary_table.sort_index()
# adding p values
p_values = reg2.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table

# interpreting coefficients in PD model
# grade variable - summary table has grade A-F, G is refrence, from Woe A is best and G is worst, A has highest coeff and f has lowest(A has highest prob of good, f has lowest, g even lower)
# what is odds of D being better than G odds(y=1|grade D)/odds(y=1|grade G) = exp^coeff grade A = 1.65, so the odds of borrower with rating D being good is 1.65 times higher than odds of borrower with rating G being good. now since the refrence category G has odds of 0(cuz its not in reg) we can compare odds of grades A-F with each other. so the odds of A to be good than b is simply = exp^coeff of A/coeff of b = exp^coeff of A-coeff of b


# validation of the model using the test data

# preparing test data

# Variables- all variables including the refrence variable, excluding categories delinq_2yrs, open_acc, pub_rec, total_rev_hi_lim, total_acc
inputs_test_with_ref_cat = loan_data_inputs_test.loc[:, ['grade:A',
                                                         'grade:B',
                                                         'grade:C',
                                                         'grade:D',
                                                         'grade:E',
                                                         'grade:F',
                                                         'grade:G',
                                                         'home_ownership:RENT_OTHER_NONE_ANY',
                                                         'home_ownership:OWN',
                                                         'home_ownership:MORTGAGE',
                                                         'addr_state:ND_NE_IA_NV_FL_HI_AL',
                                                         'addr_state:NM_VA',
                                                         'addr_state:NY',
                                                         'addr_state:OK_TN_MO_LA_MD_NC',
                                                         'addr_state:CA',
                                                         'addr_state:UT_KY_AZ_NJ',
                                                         'addr_state:AR_MI_PA_OH_MN',
                                                         'addr_state:RI_MA_DE_SD_IN',
                                                         'addr_state:GA_WA_OR',
                                                         'addr_state:WI_MT',
                                                         'addr_state:TX',
                                                         'addr_state:IL_CT',
                                                         'addr_state:KS_SC_CO_VT_AK_MS',
                                                         'addr_state:WV_NH_WY_DC_ME_ID',
                                                         'verification_status:Not Verified',
                                                         'verification_status:Source Verified',
                                                         'verification_status:Verified',
                                                         'purpose:educ__sm_b__wedd__ren_en__mov__house',
                                                         'purpose:credit_card',
                                                         'purpose:debt_consolidation',
                                                         'purpose:oth__med__vacation',
                                                         'purpose:major_purch__car__home_impr',
                                                         'initial_list_status:f',
                                                         'initial_list_status:w',
                                                         'term_36',
                                                         'term_60',
                                                         'emp_length:0',
                                                         'emp_length:1',
                                                         'emp_length:2-4',
                                                         'emp_length:5-6',
                                                         'emp_length:7-9',
                                                         'emp_length:10',
                                                         'mths_since_issue_d:<38',
                                                         'mths_since_issue_d:<38-39',
                                                         'mths_since_issue_d:<40-41',
                                                         'mths_since_issue_d:<42-48',
                                                         'mths_since_issue_d:<49-52',
                                                         'mths_since_issue_d:<53-64',
                                                         'mths_since_issue_d:<65-84',
                                                         'mths_since_issue_d:>84',
                                                         'int_rate:<9.548',
                                                         'int_rate:9.548-12.025',
                                                         'int_rate:12.025-15.74',
                                                         'int_rate:15.74-20.281',
                                                         'int_rate:>20.281',
                                                         'mths_since_earliest_cr_line:<140',
                                                         'mths_since_earliest_cr_line:141-164',
                                                         'mths_since_earliest_cr_line:165-247',
                                                         'mths_since_earliest_cr_line:248-270',
                                                         'mths_since_earliest_cr_line:271-352',
                                                         'mths_since_earliest_cr_line:>352',
                                                         'inq_last_6mths:0',
                                                         'inq_last_6mths:1-2',
                                                         'inq_last_6mths:3-6',
                                                         'inq_last_6mths:>6',
                                                         'acc_now_delinq:0',
                                                         'acc_now_delinq:>=1',
                                                         'annual_inc:<20K',
                                                         'annual_inc:20K-30K',
                                                         'annual_inc:30K-40K',
                                                         'annual_inc:40K-50K',
                                                         'annual_inc:50K-60K',
                                                         'annual_inc:60K-70K',
                                                         'annual_inc:70K-80K',
                                                         'annual_inc:80K-90K',
                                                         'annual_inc:90K-100K',
                                                         'annual_inc:100K-120K',
                                                         'annual_inc:120K-140K',
                                                         'annual_inc:>140K',
                                                         'dti:<=1.4',
                                                         'dti:1.4-3.5',
                                                         'dti:3.5-7.7',
                                                         'dti:7.7-10.5',
                                                         'dti:10.5-16.1',
                                                         'dti:16.1-20.3',
                                                         'dti:20.3-21.7',
                                                         'dti:21.7-22.4',
                                                         'dti:22.4-35',
                                                         'dti:>35',
                                                         'mths_since_last_delinq:Missing',
                                                         'mths_since_last_delinq:0-3',
                                                         'mths_since_last_delinq:4-30',
                                                         'mths_since_last_delinq:31-56',
                                                         'mths_since_last_delinq:>=57',
                                                         'mths_since_last_record:Missing',
                                                         'mths_since_last_record:0-2',
                                                         'mths_since_last_record:3-20',
                                                         'mths_since_last_record:21-31',
                                                         'mths_since_last_record:32-80',
                                                         'mths_since_last_record:81-86',
                                                         'mths_since_last_record:>86',
                                                         ]]

# all reference var excluding reference variables for delinq_2yrs, open_acc, pub_rec, total_rev_hi_lim, total_acc

ref_categories = ['grade:G',
                  'home_ownership:RENT_OTHER_NONE_ANY',
                  'addr_state:ND_NE_IA_NV_FL_HI_AL',
                  'verification_status:Verified',
                  'purpose:educ__sm_b__wedd__ren_en__mov__house',
                  'initial_list_status:f',
                  'term_60',
                  'emp_length:0',
                  'mths_since_issue_d:>84',
                  'int_rate:>20.281',
                  'mths_since_earliest_cr_line:<140',
                  'inq_last_6mths:>6',
                  'acc_now_delinq:0',
                  'annual_inc:<20K',
                  'dti:>35',
                  'mths_since_last_delinq:0-3',
                  'mths_since_last_record:0-2']

inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis=1)
inputs_test.head()

# this line of code takes the model stored in reg2 to predict the outputs based on only inputs in test data as its agruments, the default cutoff is .5% of the odds = e^b0+b1x1.. > .5--> good borrower, this is default, we can change cutoff

y_hat_test = reg2.model.predict(inputs_test)
y_hat_test  # this will be an array of 1 and 0, 1 indicating good ,0 indicating bad

# say we consider that predicted prob below .3 should be classified as bad(we can select any cutoff between 0 and 1)

# this is give y_hat_test_proba as array of arrays, each element will have array of two value [PD,1-PD] prob of default and prob of no default
# the predict method returns the predicted class while the predict_proba method returns the probabilities
y_hat_test_proba = reg2.model.predict_proba(inputs_test)

# now we are overwrting the y_hat_test_proba array with only the second elements 1-PD
y_hat_test_proba = y_hat_test_proba[:, 1]
y_hat_test_proba


# here we are making a df df_actual_predicted_probs that has the targets of the test data(actual default data) and prob predicated by the model y_hat_test_proba (prob of being good)
loan_data_targets_test_temp = loan_data_targets_test
# reset_index(drop=True)drop=True means do not keep the current index as a new column. If drop=False was used, the existing index would become a column in the DataFrame inplace=True modifies the original DataFrame directly. If set to False, the function would return a new DataFrame with the reset index, leaving the original DataFrame unchanged.
loan_data_targets_test_temp.reset_index(drop=True, inplace=True)
df_actual_predicted_probs = pd.concat(
    [loan_data_targets_test_temp, pd.DataFrame(y_hat_test_proba)], axis=1)
df_actual_predicted_probs.shape


df_actual_predicted_probs.columns = [
    'loan_data_targets_test', 'y_hat_test_proba']
df_actual_predicted_probs.index = loan_data_inputs_test.index
df_actual_predicted_probs.head()


# now we check how well the model performed
# accuracy and AREA UNDER THE CURVE
# we take a prob of 50% and will check of the model's predicted prob >.5 actually matches the target(actual default data)

tr = 0.5
df_actual_predicted_probs['y_hat_test'] = np.where(
    df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)

df_actual_predicted_probs.head()

# confusion matrix-in notes- tells TP, FP, TN, FN, Accuracy = TP/TP+FP, sensitivity = TP/ TP+FN
# crosstab is a function in pandas that computes a cross-tabulation (also called a contingency table) of two (or more) categorical variables. It counts the occurrences of the combinations of values in the input columns and presents the result in a tabular format.# df_actual_predicted_probs['loan_data_targets_test']:
# This represents the actual target values from your test dataset. It's assumed that this column contains the true labels (the ground truth) for each sample, typically denoted as y_true in classification problems. These values are what the model was trying to predict.

pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'],
            df_actual_predicted_probs['y_hat_test'], rownames=['actual'], colnames=['Predicted'])

# converted in to percentages so we get TP/TP+NP+FP+FN, TN/TP+NP+FP+FN, FP/TP+NP+FP+FN, FN/TP+NP+FP+FN
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames=[
            'actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]

# Accuracy- here we get TP/TP+NP+FP+FN + TN/TP+NP+FP+FN which is accuracy
(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames=['actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (
    pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames=['actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]

# so far we used treshhold of .5(PD<.5=good, >.5=bad) and when we observed the confusion matrix we fund that true positives are high but true negatives are very low, basically model is classifying most as good because very less people have defaulted in the training dataset. we can try with various thresholds and choose that best works for us, we dont want defaults but we also want to give loans, so we use the ROC curve

# ploting ROC curve

roc_curve(df_actual_predicted_probs['loan_data_targets_test'],
          df_actual_predicted_probs['y_hat_test_proba'])

# this returns three arrays in this order False positive Rate, True Positive Rate, Thresholds

# putting these into an array
fpr, tpr, thresholds = roc_curve(
    df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

# plotting
sns.set()

plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle='--', color='k')
plt.xlabel('Flase Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# area under ROC is called AUC(area under the curve)
AUROC = roc_auc_score(
    df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
AUROC  # we got are under ROC 70.1%

# Evaluation of the model performance using two mothods: ginni coefficient and kolmogorov-smirnov coefficient

# to cal. the GC and KSC we first nees the order the PD in ascending order

df_actual_predicted_probs = df_actual_predicted_probs.sort_values(
    'y_hat_test_proba')

df_actual_predicted_probs.head()
df_actual_predicted_probs.tail()

# calculating the cumulative prob
df_actual_predicted_probs = df_actual_predicted_probs.reset_index()
df_actual_predicted_probs.head()

# we need to cumulative % of total population, the cumulative % of good borrowers, the cummulative % of bad borrowers.
df_actual_predicted_probs['cumulative N population'] = df_actual_predicted_probs.index + 1
# cumsum is for cumulative sum
df_actual_predicted_probs['cumulative N good'] = df_actual_predicted_probs['loan_data_targets_test'].cumsum()
# here we are trying to calculate the cumulative sum of bad borrowers but bad borrowers are represents as 0 so cant add them, but we can get it by cumulative total population - cumulative good borrowers
df_actual_predicted_probs['cumulative N bad'] = df_actual_predicted_probs['cumulative N population'] - \
    df_actual_predicted_probs['loan_data_targets_test'].cumsum()
df_actual_predicted_probs.head()

# calculating cumulative %
df_actual_predicted_probs['cumulative prec population'] = df_actual_predicted_probs['cumulative N population'] / (
    df_actual_predicted_probs.shape[0])
df_actual_predicted_probs['cumulative prec good'] = df_actual_predicted_probs['cumulative N good'] / \
    df_actual_predicted_probs['loan_data_targets_test'].sum()
df_actual_predicted_probs['cumulative prec bad'] = df_actual_predicted_probs['cumulative N bad'] / (
    df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['loan_data_targets_test'].sum())

# now we can plot GC and TSC
plt.plot(df_actual_predicted_probs['cumulative prec population'],
         df_actual_predicted_probs['cumulative prec bad'])
plt.plot(df_actual_predicted_probs['cumulative prec population'],
         df_actual_predicted_probs['cumulative prec population'], linestyle='--', color='k')
plt.xlabel('cumulative % Population')
plt.ylabel('cumulative % Bad')
plt.title('Gini coefficient')

# Gini = AUROC*2 -1
Gini = AUROC*2 - 1
Gini

# plotting KSC
plt.plot(df_actual_predicted_probs['y_hat_test_proba'],
         df_actual_predicted_probs['cumulative prec bad'], color='r')
plt.plot(df_actual_predicted_probs['y_hat_test_proba'],
         df_actual_predicted_probs['cumulative prec good'], color='b')
plt.xlabel('Estimated Probability of being good')
plt.ylabel('cumulative %')
plt.title('Kolmogorov-smirnov')

# KSC is the max difference between the red and the blue curve
KS = max(df_actual_predicted_probs['cumulative prec bad'] -
         df_actual_predicted_probs['cumulative prec good'])
KS
# we get the value of KS .29, the two cumulative distribution functions are sufficiently far away from each other and the model has satisfactory predictive power.


# Applying the PD model for a borrower, prob of not default 1-PD = e^sum of coeffients(where variable value is 1) / 1 + e^sum of coeffients(where variable value is 1)
# banks use scorecards to do the same for non technical people

# creating a scorecard - it is similar to applying the PD model
# we had saved the coeff for dummy variables in summary_table, except for the reference categories
# formulas that will be used in scorecard to cal score of intercept and other variables
# intercept score = [((intercept coefficient - min score)/(max sum of coeffiencts when most are 1 - minimum sum of coeffcients when most are 0 )) *(max score - min score)]+ min score     here min score is 350, max score is 800
# variable score = variable coefficient * [(max score - min score ) / (max sum of coeffiencts when most are 1 - minimum sum of coeffcients when most are 0) ]

summary_table
ref_categories

df_ref_categories = pd.DataFrame(ref_categories, columns=['Feature name'])
# We create a new dataframe with one column. Its values are the values from the 'reference_categories' list.
# We name it 'Feature name'.
df_ref_categories['Coefficients'] = 0
# We create a second column, called 'Coefficients', which contains only 0 values.
df_ref_categories['p_values'] = np.nan
# We create a third column, called 'p_values', with contains only NaN values.
df_ref_categories

df_scorecard = pd.concat([summary_table, df_ref_categories])
# Concatenates two dataframes.
df_scorecard = df_scorecard.reset_index()
# We reset the index of a dataframe.
df_scorecard

df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(
    ':').str[0]
# We create a new column, called 'Original feature name', which contains the value of the 'Feature name' column,
# up to the column symbol.

df_scorecard

min_score = 300
max_score = 850

df_scorecard.groupby('Original feature name')['Coefficients'].min()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.

min_sum_coef = df_scorecard.groupby('Original feature name')[
    'Coefficients'].min().sum()
# Up to the 'min()' method everything is the same as in te line above.
# Then, we aggregate further and sum all the minimum values.
min_sum_coef

df_scorecard.groupby('Original feature name')['Coefficients'].max()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.

max_sum_coef = df_scorecard.groupby('Original feature name')[
    'Coefficients'].max().sum()
# Up to the 'min()' method everything is the same as in te line above.
# Then, we aggregate further and sum all the maximum values.
max_sum_coef

df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * \
    (max_score - min_score) / (max_sum_coef - min_sum_coef)
# We multiply the value of the 'Coefficients' column by the ration of the differences between
# maximum score and minimum score and maximum sum of coefficients and minimum sum of cefficients.
df_scorecard

df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (
    max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
# We divide the difference of the value of the 'Coefficients' column and the minimum sum of coefficients by
# the difference of the maximum sum of coefficients and the minimum sum of coefficients.
# Then, we multiply that by the difference between the maximum score and the minimum score.
# Then, we add minimum score.
df_scorecard

df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
# We round the values of the 'Score - Calculation' column.
df_scorecard

min_sum_score_prel = df_scorecard.groupby('Original feature name')[
    'Score - Preliminary'].min().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
# Sums all minimum values.
min_sum_score_prel

max_sum_score_prel = df_scorecard.groupby('Original feature name')[
    'Score - Preliminary'].max().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.
# Sums all maximum values.
max_sum_score_prel

# # One has to be added from the maximum score for one original variable. Which one? We'll evaluate based on differences.

df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - \
    df_scorecard['Score - Calculation']
df_scorecard

df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
df_scorecard['Score - Final'][77] = 16
df_scorecard

min_sum_score_prel = df_scorecard.groupby('Original feature name')[
    'Score - Final'].min().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
# Sums all minimum values.
min_sum_score_prel

max_sum_score_prel = df_scorecard.groupby('Original feature name')[
    'Score - Final'].max().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.
# Sums all maximum values.
max_sum_score_prel


# calculating credit scores

inputs_test_with_ref_cat.head()

df_scorecard

inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat

inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
# We insert a column in the dataframe, with an index of 0, that is, in the beginning of the dataframe.
# The name of that column is 'Intercept', and its values are 1s.

inputs_test_with_ref_cat_w_intercept.head()

inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[
    df_scorecard['Feature name'].values]
# Here, from the 'inputs_test_with_ref_cat_w_intercept' dataframe, we keep only the columns with column names,
# exactly equal to the row values of the 'Feature name' column from the 'df_scorecard' dataframe.

inputs_test_with_ref_cat_w_intercept.head()

scorecard_scores = df_scorecard['Score - Final']

inputs_test_with_ref_cat_w_intercept.shape

scorecard_scores.shape

scorecard_scores = scorecard_scores.values.reshape(102, 1)

scorecard_scores.shape

y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)
# Here we multiply the values of each row of the dataframe by the values of each column of the variable,
# which is an argument of the 'dot' method, and sum them. It's essentially the sum of the products.

# we can also calculate PD from the score


sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)
                       ) * (max_sum_coef - min_sum_coef) + min_sum_coef
# We divide the difference between the scores and the minimum score by
# the difference between the maximum score and the minimum score.
# Then, we multiply that by the difference between the maximum sum of coefficients and the minimum sum of coefficients.
# Then, we add the minimum sum of coefficients.


sum_coef_from_score = sum_coef_from_score.apply(
    pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
# Handle NaN values (e.g., replacing with 0)
sum_coef_from_score.fillna(0, inplace=True)

y_hat_proba_from_score = np.exp(
    sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)


# Here we divide an exponent raised to sum of coefficients from score by
# an exponent raised to sum of coefficients from score plus one.
y_hat_proba_from_score.head()

y_hat_test_proba[0: 5]
df_actual_predicted_probs['y_hat_test_proba'].head()

# setting cutoff- at what level of PD or score should we approve the loan, if we choose a high cutoff then we give less but high quality loans if we choose a low cut off then we give more but low quality loans


# We need the confusion matrix again.
# np.where(np.squeeze(np.array(loan_data_targets_test)) == np.where(y_hat_test_proba >= tr, 1, 0), 1, 0).sum() / loan_data_targets_test.shape[0]
tr = 0.9
df_actual_predicted_probs['y_hat_test'] = np.where(
    df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)
# df_actual_predicted_probs['loan_data_targets_test'] == np.where(df_actual_predicted_probs['y_hat_test_proba'] >= tr, 1, 0)

pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'],
            df_actual_predicted_probs['y_hat_test'], rownames=['Actual'], colnames=['Predicted'])

pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames=[
            'Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]

(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (
    pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]


roc_curve(df_actual_predicted_probs['loan_data_targets_test'],
          df_actual_predicted_probs['y_hat_test_proba'])

fpr, tpr, thresholds = roc_curve(
    df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

sns.set()

plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle='--', color='k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

thresholds
thresholds.shape

df_cutoffs = pd.concat(
    [pd.DataFrame(thresholds), pd.DataFrame(fpr), pd.DataFrame(tpr)], axis=1)
# We concatenate 3 dataframes along the columns.

df_cutoffs.columns = ['thresholds', 'fpr', 'tpr']
# We name the columns of the dataframe 'thresholds', 'fpr', and 'tpr'.

df_cutoffs.head()

df_cutoffs['thresholds'][0] = 1 - 1 / np.power(10, 16)
# Let the first threshold (the value of the thresholds column with index 0) be equal to a number, very close to 1
# but smaller than 1, say 1 - 1 / 10 ^ 16.

df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) -
                       min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()
# The score corresponsing to each threshold equals:
# The the difference between the natural logarithm of the ratio of the threshold and 1 minus the threshold and
# the minimum sum of coefficients
# multiplied by
# the sum of the minimum score and the ratio of the difference between the maximum score and minimum score and
# the difference between the maximum sum of coefficients and the minimum sum of coefficients.

df_cutoffs.head()

df_cutoffs['Score'][0] = max_score

df_cutoffs.head()

df_cutoffs.tail()

# We define a function called 'n_approved' which assigns a value of 1 if a predicted probability
# is greater than the parameter p, which is a threshold, and a value of 0, if it is not.
# Then it sums the column.
# Thus, if given any percentage values, the function will return
# the number of rows wih estimated probabilites greater than the threshold.


def n_approved(p):
    return np.where(df_actual_predicted_probs['y_hat_test_proba'] >= p, 1, 0).sum()


df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)
# Assuming that all credit applications above a given probability of being 'good' will be approved,
# when we apply the 'n_approved' function to a threshold, it will return the number of approved applications.
# Thus, here we calculate the number of approved appliations for al thresholds.
df_cutoffs['N Rejected'] = df_actual_predicted_probs['y_hat_test_proba'].shape[0] - \
    df_cutoffs['N Approved']
# Then, we calculate the number of rejected applications for each threshold.
# It is the difference between the total number of applications and the approved applications for that threshold.
df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / \
    df_actual_predicted_probs['y_hat_test_proba'].shape[0]
# Approval rate equalts the ratio of the approved applications and all applications.
df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']
# Rejection rate equals one minus approval rate.

df_cutoffs.head()
df_cutoffs.tail()

df_cutoffs.iloc[5000: 6200, ]
# Here we display the dataframe with cutoffs form line with index 5000 to line with index 6200.

df_cutoffs.iloc[1000: 2000, ]
# Here we display the dataframe with cutoffs form line with index 1000 to line with index 2000.

inputs_train_with_ref_cat.to_csv('inputs_train_with_ref_cat.csv')

df_scorecard.to_csv('df_scorecard.csv')


pickle.dump(reg2, open('pd_model.sav', 'wb'))
