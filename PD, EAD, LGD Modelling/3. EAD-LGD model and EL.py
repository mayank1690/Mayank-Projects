# LGD EAD DATA PREP

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import scipy.stats as stat
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

loan_data_preprocessed_backup = pd.read_csv(
    'C:\\Users\\mayan\\Downloads\\loan_data_2007_2014_preprocessed.csv')
loan_data_preprocessed = loan_data_preprocessed_backup

# to build the LGD(% of unrecovered debt after the default) and EAD($ amount of exposure at time of debt) model we need to find the % of exposure that was lost after the borrower defaulted (LGD), the amount of exposure at the moment the borrower defaulted (EAD), this data exists only for the accounts that defaulted, it would not be right to include all of the defaulted accounts in modelling LGD EAD. because a part of the debt might have been recovered given enough time to the borrower. when building LGD EAD model, it is a good practice to build models with data from borrowers that have had enough time to repay part of the remaining debt(i.e defaulted and written off), in our data these are only the accounts that have been written off(loan_status: charged_off and does not meet credit policy.status:charged off)

loan_data_preprocessed.columns.values
loan_data_preprocessed['loan_status'].unique()

loan_data_defaults = loan_data_preprocessed[loan_data_preprocessed['loan_status'].isin(
    ['Charged Off', 'Does not meet the credit policy. Status:Charged Off'])]
loan_data_defaults.shape

# independent var for ead lgd model will be same as indipendent var for PD model

loan_data_defaults.isnull().sum()

# the only var with missing values, that needs to be handled is mths_since_last_delinq, mths_since_last_record, earlier(PD modek) we had created a seperate dummy of where these values were missing, and then used woe to see impact, now for discrete variables we are going to create as many dummy variables as the number of categories and for continuous variables we can use them as they are or transform them, there is no need for fine classing or coarse classing.

# we will impute missing values in mths_since_last_delinq, mths_since_last_record, as 0
loan_data_defaults['mths_since_last_delinq'].fillna(0, inplace=True)
loan_data_defaults['mths_since_last_record'].fillna(0, inplace=True)

loan_data_defaults['mths_since_last_delinq'].isna().sum()
loan_data_defaults['mths_since_last_record'].isna().sum()

# dependent variable for LGD(% of exposure not recovered) - the approach is to model the proportion of exposure that has been recovered after the default(called recovery rate) LGD = 1- recovery rate
# funded amount has total amount that was lost the moment borrower defaulted, recoveries col has amount recovered after defualt
# recovery rate = recoveries/funded amount

loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / \
    loan_data_defaults['funded_amnt']

loan_data_defaults['recovery_rate'].describe()

# max recovery rate shown as 1.22 which is illogical, we can truncate(cast) the values outside the range 0 to 1

loan_data_defaults['recovery_rate'] = np.where(
    loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
loan_data_defaults['recovery_rate'] = np.where(
    loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])

# dependent variable for EAD - the total funded amount amt is the utilised limit (disbursed amt) at the time of defualt  EAD = total funded amount  * Credit conversion factor, CCF tells the proportion of unrecovered principal

# 'total recovered principal' reflects the total payments made on the principal of the loan, so funded amt(disbursed) - total recovered principal is current exposure


loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'] -
                             loan_data_defaults['total_rec_prncp']) / loan_data_defaults['funded_amnt']

loan_data_defaults['CCF'].describe()

loan_data_defaults.to_csv('loan_data_defaults.csv')


sns.set()

plt.hist(loan_data_defaults['recovery_rate'], bins=50)
plt.hist(loan_data_defaults['CCF'], bins=100)

# recovery rate and CCF are proportions bounded by 0 and 1, the density of proportions is best describes as a specific distribution called beta distribution, the regrssion model used to asses the impact of independent variables on a beta distribution is called deta regression

# python has no good library for beta regression but R has, so we will not use beta regression in this model

# from the plots we found that more than half obs have recovery rate of 0 so we will have two stage approach
# 1. is the recovery rate 0 or greater than 0- logistic regression
# 2. if the recovery rate is greater than 0 than how much is it?

loan_data_defaults['recovery_rate_0_1'] = np.where(
    loan_data_defaults['recovery_rate'] == 0, 0, 1)

# so our approach is going to be we will use a logistic regression to predict if the recovery rate is going to be 0 or greter than 0, if it is greater than 0 we will use a linear reg to see how much is it going to be.

# to predict CCF we will use a linear regression model


# LGD model
# first we create log reg to predict if recovery rate is 0 or >0

# splitting data into train and test


lgd_inputs_stage_1_train, lgd_inputs_stage_1_test, lgd_targets_stage_1_train, lgd_targets_stage_1_test = train_test_split(loan_data_defaults.drop(
    ['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis=1), loan_data_defaults['recovery_rate_0_1'], test_size=0.2, random_state=42)

# preparing inputs
# we will put all relevant features in a df, remove reference cat(to avoid multicollinearity), do log reg, get p values from our own method,

features_all = ['grade:A',
                'grade:B',
                'grade:C',
                'grade:D',
                'grade:E',
                'grade:F',
                'grade:G',
                'home_ownership:MORTGAGE',
                'home_ownership:NONE',
                'home_ownership:OTHER',
                'home_ownership:OWN',
                'home_ownership:RENT',
                'verification_status:Not Verified',
                'verification_status:Source Verified',
                'verification_status:Verified',
                'purpose:car',
                'purpose:credit_card',
                'purpose:debt_consolidation',
                'purpose:educational',
                'purpose:home_improvement',
                'purpose:house',
                'purpose:major_purchase',
                'purpose:medical',
                'purpose:moving',
                'purpose:other',
                'purpose:renewable_energy',
                'purpose:small_business',
                'purpose:vacation',
                'purpose:wedding',
                'initial_list_status:f',
                'initial_list_status:w',
                'term_int',
                'emp_length_int',
                'mths_since_issue_d',
                'mths_since_earliest_cr_line',
                'funded_amnt',
                'int_rate',
                'installment',
                'annual_inc',
                'dti',
                'delinq_2yrs',
                'inq_last_6mths',
                'mths_since_last_delinq',
                'mths_since_last_record',
                'open_acc',
                'pub_rec',
                'total_acc',
                'acc_now_delinq',
                'total_rev_hi_lim']
# List of all independent variables for the models.


features_reference_cat = ['grade:G',
                          'home_ownership:RENT',
                          'verification_status:Verified',
                          'purpose:credit_card',
                          'initial_list_status:f']
# List of the dummy variable reference categories.


lgd_inputs_stage_1_train = lgd_inputs_stage_1_train[features_all]
lgd_inputs_stage_1_train = lgd_inputs_stage_1_train.drop(
    features_reference_cat, axis=1)
lgd_inputs_stage_1_train.isnull().sum()

# estimating the model

# Class to display p-values for logistic regression in sklearn.


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
        # self.z_scores = z_scores
        self.p_values = p_values
        # self.sigma_estimates = sigma_estimates
        # self.F_ij = F_ij


reg_lgd_st_1 = LogisticRegression_with_p_values()
# fitting the model
reg_lgd_st_1.fit(lgd_inputs_stage_1_train, lgd_targets_stage_1_train)
feature_name = lgd_inputs_stage_1_train.columns.values

# creating the summary table, just like in PD model
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg_lgd_st_1.coef_)
summary_table.index = summary_table.index + 1
# Increases the index of every row of the dataframe with 1.
summary_table.loc[0] = ['Intercept', reg_lgd_st_1.intercept_[0]]
# Assigns values of the row with index 0 of the dataframe.
summary_table = summary_table.sort_index()
p_values = reg_lgd_st_1.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table


# testing the model
# preparing test df
lgd_inputs_stage_1_test = lgd_inputs_stage_1_test[features_all]
lgd_inputs_stage_1_test = lgd_inputs_stage_1_test.drop(
    features_reference_cat, axis=1)

# applying the model
y_hat_test_lgd_stage_1 = reg_lgd_st_1.model.predict(lgd_inputs_stage_1_test)

y_hat_test_lgd_stage_1

# predicting probabilities
y_hat_test_proba_lgd_stage_1 = reg_lgd_st_1.model.predict_proba(
    lgd_inputs_stage_1_test)

y_hat_test_proba_lgd_stage_1

# keeping only the prob of second column, i.e the prob of recovery rate being 1
y_hat_test_proba_lgd_stage_1 = y_hat_test_proba_lgd_stage_1[:][:, 1]

y_hat_test_proba_lgd_stage_1

lgd_targets_stage_1_test_temp = lgd_targets_stage_1_test

lgd_targets_stage_1_test_temp.reset_index(drop=True, inplace=True)

df_actual_predicted_probs = pd.concat(
    [lgd_targets_stage_1_test_temp, pd.DataFrame(y_hat_test_proba_lgd_stage_1)], axis=1)

df_actual_predicted_probs.columns = [
    'lgd_targets_stage_1_test', 'y_hat_test_proba_lgd_stage_1']

df_actual_predicted_probs.index = lgd_inputs_stage_1_test.index

df_actual_predicted_probs.head()


# estimating the accuracy of the model, just like we did in PD model, we will create confusion matrix, ROC, cal AUROC

tr = 0.5

df_actual_predicted_probs['y_hat_test_lgd_stage_1'] = np.where(
    df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1'] > tr, 1, 0)

pd.crosstab(df_actual_predicted_probs['lgd_targets_stage_1_test'],
            df_actual_predicted_probs['y_hat_test_lgd_stage_1'], rownames=['Actual'], colnames=['Predicted'])

pd.crosstab(df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_lgd_stage_1'], rownames=[
            'Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]

# Here we calculate Accuracy of the model, which is the sum of the diagonal rates.
(pd.crosstab(df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_lgd_stage_1'], rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (
    pd.crosstab(df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_lgd_stage_1'], rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]


# ROC and AUROC

fpr, tpr, thresholds = roc_curve(
    df_actual_predicted_probs['lgd_targets_stage_1_test'], df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1'])

plt.plot(fpr, tpr)

plt.plot(fpr, fpr, linestyle='--', color='k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

AUROC = roc_auc_score(df_actual_predicted_probs['lgd_targets_stage_1_test'],
                      df_actual_predicted_probs['y_hat_test_proba_lgd_stage_1'])
AUROC

# saving the model- we use pickle to save and load the model, .dump to save, arguments- object to be dumped, filename, wb-writebytes


pickle.dump(reg_lgd_st_1, open('lgd_model_stage_1.sav', 'wb'))


# STAGE 2 - we have made a model to predict wheather the recovery rate is 0 or greater than 0, now we can move to the next stage i.e linear reg to predict the recovery rate if it greater than 0, i,e not 0

# we coded recovery rate as 0 and 1 in recovery rate column, extract where value is 1

lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]
lgd_stage_2_data.head()

# splitting data into test and train
lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop(
    ['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis=1), lgd_stage_2_data['recovery_rate'], test_size=0.2, random_state=42)

# i have modified the code to get p values in a different way
lgd_inputs_stage_2_train = lgd_inputs_stage_2_train[features_all]
lgd_inputs_stage_2_train = lgd_inputs_stage_2_train.drop(
    features_reference_cat, axis=1)

reg_lgd_st_2 = LinearRegression()
reg_lgd_st_2.fit(lgd_inputs_stage_2_train, lgd_targets_stage_2_train)

# Add a constant (intercept) to the input features for statsmodels
X_with_const = sm.add_constant(lgd_inputs_stage_2_train)

# Fit the model using statsmodels to get p-values
model = sm.OLS(lgd_targets_stage_2_train, X_with_const)
results = model.fit()

# Create the summary table
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg_lgd_st_2.coef_)

# Insert the intercept row at the beginning
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_lgd_st_2.intercept_]
summary_table = summary_table.sort_index()

# Get p-values from the statsmodels results
p_values = results.pvalues.values
# Ensure the p-values array includes the intercept (add NaN to the start)
# p_values = np.insert(p_values, 0, np.nan)  # Insert NaN for intercept
summary_table['p_values'] = p_values.round(3)
summary_table.loc[0, 'p_values'] = np.nan

print(summary_table)


# linear regression model evaluation- how well it predicts recovery rates that are greater than 0, here instead of confusion matrix we will use correlation matrix

# preparing test data
lgd_inputs_stage_2_test = lgd_inputs_stage_2_test[features_all]
lgd_inputs_stage_2_test = lgd_inputs_stage_2_test.drop(
    features_reference_cat, axis=1)
lgd_inputs_stage_2_test.columns.values

# predicting using the model
y_hat_test_lgd_stage_2 = reg_lgd_st_2.predict(lgd_inputs_stage_2_test)

# creating a temp df with actual and predicted recovery rate
lgd_targets_stage_2_test_temp = lgd_targets_stage_2_test
lgd_targets_stage_2_test_temp = lgd_targets_stage_2_test_temp.reset_index(
    drop=True)

# calculationg correlation between actual and predicted
pd.concat([lgd_targets_stage_2_test_temp, pd.DataFrame(
    y_hat_test_lgd_stage_2)], axis=1).corr()

# we will also plot the residuals and check if they are normally distributed around the mean 0
sns.distplot(lgd_targets_stage_2_test - y_hat_test_lgd_stage_2)
# yes they are- indicating it is a good model

# saving the model
pickle.dump(reg_lgd_st_2, open('lgd_model_stage_2.sav', 'wb'))


# combining the two models
y_hat_test_lgd_stage_2_all = reg_lgd_st_2.predict(lgd_inputs_stage_1_test)

y_hat_test_lgd_stage_2_all  # this will give us recovery rate of all observations and y_hat_test_lgd_stage_1 will give return 0 for entries predicted to have RR 0 and 1 for entries predicted to have RR>0 we can multiply the two to comnine the model

# combiming the models
y_hat_test_lgd = y_hat_test_lgd_stage_1 * y_hat_test_lgd_stage_2_all

pd.DataFrame(y_hat_test_lgd).describe()

# we found that the min value of RR is -ve so we bound it by 0 and 1

y_hat_test_lgd = np.where(y_hat_test_lgd < 0, 0, y_hat_test_lgd)
y_hat_test_lgd = np.where(y_hat_test_lgd > 1, 1, y_hat_test_lgd)

pd.DataFrame(y_hat_test_lgd).describe()


# EAD MODEL

# dividing data into test and train dataset
ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop(
    ['good_bad', 'recovery_rate', 'recovery_rate_0_1', 'CCF'], axis=1), loan_data_defaults['CCF'], test_size=0.2, random_state=42)

ead_inputs_train.columns.values

ead_inputs_train = ead_inputs_train[features_all]

ead_inputs_train = ead_inputs_train.drop(features_reference_cat, axis=1)


# fitting the lin reg model
reg_ead = LinearRegression()
reg_ead.fit(ead_inputs_train, ead_targets_train)


feature_name = ead_inputs_train.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg_ead.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_ead.intercept_]
summary_table = summary_table.sort_index()

X_with_const = sm.add_constant(ead_inputs_train)
# Fit the model using statsmodels to get p-values
model = sm.OLS(ead_targets_train, X_with_const)
results = model.fit()


p_values = results.pvalues.values

summary_table['p_values'] = p_values
summary_table


# model validation

# preparing data
ead_inputs_test = ead_inputs_test[features_all]
ead_inputs_test = ead_inputs_test.drop(features_reference_cat, axis=1)
ead_inputs_test.columns.values

# calculating predicted values
y_hat_test_ead = reg_ead.predict(ead_inputs_test)

ead_targets_test_temp = ead_targets_test
ead_targets_test_temp = ead_targets_test_temp.reset_index(drop=True)

pd.concat([ead_targets_test_temp, pd.DataFrame(y_hat_test_ead)], axis=1).corr()

sns.distplot(ead_targets_test - y_hat_test_ead)

pd.DataFrame(y_hat_test_ead).describe()

# bounding ccf by 1 and 0
y_hat_test_ead = np.where(y_hat_test_ead < 0, 0, y_hat_test_ead)
y_hat_test_ead = np.where(y_hat_test_ead > 1, 1, y_hat_test_ead)

pd.DataFrame(y_hat_test_ead).describe()


# Expected loss

# EL - PD * LGD * EAD
# we'll calculate EL of the bank not the indi custusomers, EL of bank = sum of EL of indi customers

# preparing data
loan_data_preprocessed.head()

# while preparing data for ead and LGD models we imputed some column;s nan values, we need to do that again for this data
loan_data_preprocessed['mths_since_last_delinq'].fillna(0, inplace=True)
loan_data_preprocessed['mths_since_last_record'].fillna(0, inplace=True)

loan_data_preprocessed_lgd_ead = loan_data_preprocessed[features_all]
loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(
    features_reference_cat, axis=1)

# then we calculate the LGD
loan_data_preprocessed['recovery_rate_st_1'] = reg_lgd_st_1.model.predict(
    loan_data_preprocessed_lgd_ead)  # weather RR is 0 or > 0
loan_data_preprocessed['recovery_rate_st_2'] = reg_lgd_st_2.predict(
    loan_data_preprocessed_lgd_ead)  # if RR is > 0 then how much

loan_data_preprocessed['recovery_rate'] = loan_data_preprocessed['recovery_rate_st_1'] * \
    loan_data_preprocessed['recovery_rate_st_2']  # RR is stage1*stage2
# bounding RR by 0 and 1
loan_data_preprocessed['recovery_rate'] = np.where(
    loan_data_preprocessed['recovery_rate'] < 0, 0, loan_data_preprocessed['recovery_rate'])
loan_data_preprocessed['recovery_rate'] = np.where(
    loan_data_preprocessed['recovery_rate'] > 1, 1, loan_data_preprocessed['recovery_rate'])

loan_data_preprocessed['LGD'] = 1 - \
    loan_data_preprocessed['recovery_rate']  # LGD = 1- RR

loan_data_preprocessed['LGD'].describe()

# calculating EAD

# cal CCF
loan_data_preprocessed['CCF'] = reg_ead.predict(loan_data_preprocessed_lgd_ead)
# bounding CCF values by 0 and 1
loan_data_preprocessed['CCF'] = np.where(
    loan_data_preprocessed['CCF'] < 0, 0, loan_data_preprocessed['CCF'])
loan_data_preprocessed['CCF'] = np.where(
    loan_data_preprocessed['CCF'] > 1, 1, loan_data_preprocessed['CCF'])

# EAD = ccf * funded amount
loan_data_preprocessed['EAD'] = loan_data_preprocessed['CCF'] * \
    loan_data_preprocessed_lgd_ead['funded_amnt']

loan_data_preprocessed['EAD'].describe()

loan_data_preprocessed.head()


# calculating PD

# for PD we had a different set of variables, these were dummy variables- we used the had created them and split the DF in training and test df

loan_data_inputs_train = pd.read_csv('loan_data_inputs_train.csv')
loan_data_inputs_test = pd.read_csv('loan_data_inputs_test.csv')

# note we want to add the rows of both the DF hence axis is 0
loan_data_inputs_pd = pd.concat(
    [loan_data_inputs_train, loan_data_inputs_test], axis=0)


loan_data_inputs_pd.shape
loan_data_inputs_pd.head()

loan_data_inputs_pd = loan_data_inputs_pd.set_index('Unnamed: 0')
loan_data_inputs_pd.head()

features_all_pd = ['grade:A',
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
                   ]


ref_categories_pd = ['grade:G',
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


loan_data_inputs_pd_temp = loan_data_inputs_pd[features_all_pd]
loan_data_inputs_pd_temp = loan_data_inputs_pd_temp.drop(
    ref_categories_pd, axis=1)
loan_data_inputs_pd_temp.shape

# we had saved the PD model, loading it here
reg_pd = pickle.load(open('pd_model.sav', 'rb'))

# we know that predict proba return array of arrays where its predicts [[PD] , [1-PD]] for all entries ie [prob of bad borrower , prob of good borrower], we only need the PD(0 element), so we select all arrays and then all rows of all arrays and the first column
reg_pd.model.predict_proba(loan_data_inputs_pd_temp)[:][:, 0]

loan_data_inputs_pd['PD'] = reg_pd.model.predict_proba(
    loan_data_inputs_pd_temp)[:][:, 0]

loan_data_inputs_pd['PD'].head()

loan_data_inputs_pd['PD'].describe()


loan_data_preprocessed_new = pd.concat(
    [loan_data_preprocessed, loan_data_inputs_pd], axis=1)

loan_data_preprocessed_new.shape


loan_data_preprocessed_new.head()

# calulating EL = pd*lgd*ead

loan_data_preprocessed_new['EL'] = loan_data_preprocessed_new['PD'] * \
    loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['EAD']


loan_data_preprocessed_new['EL'].describe()

loan_data_preprocessed_new[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].head()

loan_data_preprocessed_new['funded_amnt'].describe()

loan_data_preprocessed_new['EL'].sum()

loan_data_preprocessed_new['funded_amnt'].sum()

loan_data_preprocessed_new['EL'].sum(
) / loan_data_preprocessed_new['funded_amnt'].sum()
