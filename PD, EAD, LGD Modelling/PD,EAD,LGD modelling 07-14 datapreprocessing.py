# PD, LGD EAD modelling using loan data from 2007-14

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

Loan_data_backup = pd.read_csv(
    "C:\\Users\\mayan\\Downloads\\loan_data_2007_2014(2).csv")
loan_data = Loan_data_backup.copy()

# pd.options.display.max_Columns = None
loan_data.head()
loan_data.tail()
loan_data.columns.values
loan_data.info()


# loan_data.iloc[:,0]
# loan_data['id'][0:10]
# df['Salary'].isna().sum()


# preprocessing data

# processing emp_lenth- convert str into numeric

loan_data['emp_length'].unique()

loan_data['emp_length_int'] = loan_data['emp_length'].str.replace(
    r'10+ years', str(10))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(
    r'< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(
    r'n/a', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(
    r' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(
    r' year', '')

loan_data['emp_length_int'].unique()
type(loan_data['emp_length_int'][0])
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])
type(loan_data['emp_length_int'][0])

# processing term- converting to numeric

loan_data['term'].unique()
loan_data['term_int'] = loan_data['term'].str.replace(r' months', '')
loan_data['term_int'].unique()
loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])
type(loan_data['term_int'][0])


# loan_data['earliest_cr_line']
# what we can do with the earliest credit line is to calculate the time that
# has passed since the credit line, we will calculate the No of months since the
# earliest credit line is issued,
# we will also do the same for no of months since the loan was issued

type(loan_data['earliest_cr_line'])
loan_data['earliest_cr_line_date'] = pd.to_datetime(
    loan_data['earliest_cr_line'], format='%b-%y', errors='coerce')
type(loan_data['earliest_cr_line_date'])

# loan_data['months_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date'])/np.timedelta64(1, 'M')))
# Define a reference date
reference_date = pd.to_datetime('2017-12-01')
# Calculate the difference in months
loan_data['mths_since_earliest_cr_line'] = (
    (reference_date.year - loan_data['earliest_cr_line_date'].dt.year) * 12 +
    (reference_date.month - loan_data['earliest_cr_line_date'].dt.month)
)
# Display the result
print(loan_data['mths_since_earliest_cr_line'])

# processing mths_since_earliest_cr_line
loan_data['mths_since_earliest_cr_line'].describe()
# minimum value of -ve which is illogical,
# extracting data only for rows where months since earliest credit line is -ve
loan_data.loc[:, ['earliest_cr_line', 'earliest_cr_line_date',
                  'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]

# due to dates starting from 1970's in python, it has taken dates older than 1970 (represented as, say jan 69) as 2069 so basically these are the oldest loans so we replace the months since earliest credit line with the max value of the columns, also since the size if small wont affect model

# loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line' < 0]] = loan_data['mths_since_earliest_cr_line'].max()

loan_data.loc[loan_data['mths_since_earliest_cr_line'] < 0,
              'mths_since_earliest_cr_line'] = loan_data['mths_since_earliest_cr_line'].max()

min(loan_data['mths_since_earliest_cr_line'])

# processing issue d
loan_data['issue_d']
loan_data['issue_d_date'] = pd.to_datetime(
    loan_data['issue_d'], format='%b-%y', errors='coerce')
loan_data['issue_d_date']
# Define a reference date
reference_date = pd.to_datetime('2017-12-01')

# Calculate the difference in months
loan_data['mths_since_issue_d'] = (
    (reference_date.year - loan_data['issue_d_date'].dt.year) * 12 +
    (reference_date.month - loan_data['issue_d_date'].dt.month)
)

# Display the result
print(loan_data['mths_since_issue_d'])

loan_data['mths_since_issue_d'].describe()

# credting dummy variables, we can use pandas pd.getdummies, note k-1 dummies are to be created eventually
pd.get_dummies(loan_data['grade'], prefix='grade', prefix_sep=':', dtype=int)
# prefix fn of get_dummies used to add prefix specifying dummies relate to which original column, dtype used to get it as 1 and 0s else it was true and false

# creating a different DF of dummies which later to be appended to makin laon_data

loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix='grade', prefix_sep=':'),
                     pd.get_dummies(
                         loan_data['sub_grade'], prefix='sub_grade', prefix_sep=':'),
                     pd.get_dummies(
                         loan_data['home_ownership'], prefix='home_ownership', prefix_sep=':'),
                     pd.get_dummies(
                         loan_data['verification_status'], prefix='verification_status', prefix_sep=':'),
                     pd.get_dummies(
                         loan_data['loan_status'], prefix='loan_status', prefix_sep=':'),
                     pd.get_dummies(
                         loan_data['purpose'], prefix='purpose', prefix_sep=':'),
                     pd.get_dummies(
                         loan_data['addr_state'], prefix='addr_state', prefix_sep=':'),
                     pd.get_dummies(
                         loan_data['initial_list_status'], prefix='initial_list_status', prefix_sep=':'),
                     ]

loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
loan_data_dummies
type(loan_data_dummies)
loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)
loan_data.columns.values

# handling missing values
loan_data.isnull()
pd.options.display.max_rows = None
loan_data.isnull().sum()
pd.options.display.max_rows = 100
# filling na values with values in column 'funded amt'
loan_data['total_rev_hi_lim'] = loan_data['total_rev_hi_lim'].fillna(
    loan_data['funded_amnt'])
loan_data['total_rev_hi_lim'].isnull().sum()
loan_data['annual_inc'] = loan_data['annual_inc'].fillna(
    loan_data['annual_inc'].mean())
loan_data['annual_inc'].isnull().sum()

loan_data['mths_since_earliest_cr_line'] = loan_data['mths_since_earliest_cr_line'].fillna(
    0)
loan_data['acc_now_delinq'] = loan_data['acc_now_delinq'].fillna(0)
loan_data['total_acc'] = loan_data['total_acc'].fillna(0)
loan_data['pub_rec'] = loan_data['pub_rec'].fillna(0)
loan_data['open_acc'] = loan_data['open_acc'].fillna(0)
loan_data['inq_last_6mths'] = loan_data['inq_last_6mths'].fillna(0)
loan_data['delinq_2yrs'] = loan_data['delinq_2yrs'].fillna(0)
loan_data['emp_length_int'] = loan_data['emp_length_int'].fillna(0)

loan_data['mths_since_earliest_cr_line'].isnull().sum()
loan_data['acc_now_delinq'].isnull().sum()
loan_data['total_acc'].isnull().sum()
loan_data['pub_rec'].isnull().sum()
loan_data['open_acc'].isnull().sum()
loan_data['inq_last_6mths'].isnull().sum()
loan_data['delinq_2yrs'].isnull().sum()
loan_data['emp_length_int'].isnull().sum()


# preprocessing data for PD modelling- converting  to dummies
# dependent variable. Good/Bad (default) definition. Default and non default accounts.
loan_data['loan_status'].unique()
loan_data['loan_status'].value_counts()
loan_data['loan_status'].value_counts()/loan_data['loan_status'].count()

# classifying bad loans as 0 and good loans as 1- we want +ve coeff to indicate good coefficient
# default definition-if account status is 'Charged Off' , 'Default', 'Does not meet the credit policy. Status:Charged Off' , 'Late (31-120 days)'

loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(
    ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']), 0, 1)

loan_data['good_bad']

# to convert continuous variables into dummy variables, we first divide them into classes(fine classing), then we combine these classes based on how similar they are(based on woe and number of observations) and so that we have lesser classes this is called coarse classing

# train_test_split splits data into training and test data, gives four arrays
# 1. train data with inputs, 2. test data with inputs 3,4. train and test data with targets
# here we are splitting the data we give the input and the target dataframes,
# in the first section we gave all var except the target and in the second we only gave the target
train_test_split(loan_data.drop('good_bad', axis=1), loan_data['good_bad'])

# we will put the four arrays in four objects the below is example, futher modified in following code
# loan_data_inputs_train, loan_data_inputs_test, loan_data_inputs_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis = 1), loan_data['good_bad'])

# another way to do it is define the size of test df, test size = .2, shuffle parameter shuffles the data, its a boolean with values true/false, but we are not shuffling here randomly to allow repliction,, we use random_state = 42 to avoid default shuffling

loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(
    loan_data.drop('good_bad', axis=1), loan_data['good_bad'], test_size=0.2, random_state=42)

loan_data_inputs_train.shape
loan_data_inputs_test.shape
loan_data_targets_train.shape
loan_data_targets_test.shape

# data pre processing splittling the data- discrete categorical data
# Here we are creating a different df for preprocessing the data, by creating a different df (df_inputs_prepr,df_targets_prepr) after pre processing the data will be returned to loan_data_inputs_train, loan_data_targets_train
# these preprocessing steps will also be applied to test data using the same following code at that time we just need to replace loan_data_inputs_train, loan_data_targets_train with loan_data_inputs_test, loan_data_targets_test

df_inputs_prepr = loan_data_inputs_train
df_targets_prepr = loan_data_targets_train

df_inputs_prepr['grade'].unique()
# we will create a dataframe containig only grades from inputs and good_bad from the target
df1 = pd.concat([df_inputs_prepr['grade'], df_targets_prepr], axis=1)
df1.head()

# now we will cal weight of evidence for grade
# groupby, we are grouping the df1 based on first column(grades) all similar grade A together, (group values(A,B) become indexes), then we also want it show the number of No observations (good and bad)in a group
# parameter of group by- DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True)
# asindex = false doesnot let group values become indexes

count_df = df1.groupby(df1.columns.values[0], as_index=False)[
    df1.columns.values[1]].count()

# we want % of good and bad for each grade to cal WOE, since good is 1 and bad is 0, we can cal % of good, and bad = 1-%good
# we can cal the good % by using mean() this will give the % of good borrowers in each grade. since there are only good and bad in second column
# mean hear in this particular case means Good(1s)+bad(0s) in a particlar grade/total values in that grade
mean_df = df1.groupby(df1.columns.values[0], as_index=False)[
    df1.columns.values[1]].mean()
df1 = pd.concat([count_df, mean_df], axis=1)
df1

df1 = df1.iloc[:, [0, 1, 3]]
print(df1.columns)

# renaming column
df1.columns = [df1.columns.values[0], 'n_obs', 'prop_good']
df1

# proportion of observation in each row
df1['prop_n_obs'] = df1['n_obs'] / df1['n_obs'].sum()
df1

df1['prop_n_obs'].sum()

df1['prop_good'].sum()

# so far df1 - n_obs(number of obs in a grade), prop_good(% of goods in a grade), prop_n_obs (No of goods in a grade/all goods in all grades)

# adding column, number of goods
df1['n_good'] = df1['prop_good'] * df1['n_obs']
# adding column, number of bads
df1['n_bad'] = (1 - df1['prop_good']) * df1['n_obs']
df1

# proportion of good borrowers = good in a category/good in all categories

df1['prop_n_good'] = df1['n_good'] / df1['n_good'].sum()
df1['prop_n_bad'] = df1['n_bad'] / df1['n_bad'].sum()
df1

# weight of evidence(for variable grade) = ln(% of good / % of bad)

df1['WoE'] = np.log(df1['prop_n_good'] / df1['prop_n_bad'])
df1

# sorting by WoE, higher up means prob of default is high

df1 = df1.sort_values(['WoE'])
df1 = df1.reset_index(drop=True)
df1

# we are trying to see how much difference is there in prop_good and WOE in adjacent rows. .diff() substracts the value in row with value in row above-The first value in the resulting Series is NaN because there is no previous value to subtract for the first row.
# The subsequent values represent the differences between consecutive rows.

df1['diff_prop_good'] = df1['prop_good'].diff().abs()
df1['diff_WoE'] = df1['WoE'].diff().abs()
df1

# now we cal the information value, IV = sum across the whole column(%good - %bad)*WoE
df1['IV'] = (df1['prop_n_good'] - df1['prop_n_bad']) * df1['WoE']
df1

# summing across the column to get IV for variable grades

df1['IV'] = df1['IV'].sum()
df1

# automating the IV calculation by defining a function that takes (training input dataframe, variable name, training output df)


def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    temp1 = df[discrete_variable_name]
    temp2 = good_bad_variable_df
    # combines independent and dependednt variable
    df = pd.concat([temp1, temp2], axis=1)
    count_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].count()
    mean_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].mean()
    df = pd.concat([count_df1, mean_df1], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    # adding column, number of goods
    df['n_good'] = df['prop_good'] * df['n_obs']
    # adding column, number of bads
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# crosschecking fn on grade variable
df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp

# importing for visulization
sns.set()

# creating a function to plot the woe, we will pass the df with WOE number, also some variable categories(a,b,c,d in grade)
# can be long so we want to rotate the variable name such that it is legible(by default value is 0, only to be passed if rotation required)


def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    # labels/categories of variable are listed in first column, also converting them to string, also converting it to np array as matplotlib performs better with np array
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weights of Evidence')
    plt.title(str('Weight of Evidence by' + df_WoE.columns[0]))
    plt.xticks(rotation=rotation_of_x_axis_labels)


plot_by_woe(df_temp)

# created a list of dummy variable in excel file loan_data_2007-2014, we have kept the category with worst WOE as a reference category, g in grade for example

# using functions on other variables
df_temp = woe_discrete(df_inputs_prepr, 'home_ownership', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# we notice that the category other, none, any have very few observation, so we combine them with the riskiets category rent
# while setting up dummy variable we want to keep categories with similat WoE together so 3 categories here 1 other+none+any+rent 2 own 3 mortgage
# below dummy var is 1 if any of the options is 1 hence all of them are combined
df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum(
    [df_inputs_prepr['home_ownership:RENT'], df_inputs_prepr['home_ownership:NONE'], df_inputs_prepr['home_ownership:OTHER'], df_inputs_prepr['home_ownership:ANY'],])

df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY']

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


df_inputs_prepr['addr_state'].unique()
df_inputs_prepr['addr_state'].nunique()

# there are 50 unique address states- well have to combine them at somepoint to reduce number of dummy variables

df_temp = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr)
df_temp
plot_by_woe(df_temp)

# northdacota not represented as no obs, setting a dummy just in case

if ['addr_state:ND'] in df_inputs_prepr.columns.values:
    pass
else:
    df_inputs_prepr['addr_state:ND'] = 0

# df_inputs_prepr['addr_state:ND']

# from the data of df_temp = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr) we observe that
# the first two states NE IA and last two states ME ID have very few observation so we add them to the 3rd and 3rd last states (worst and best based on similarities in their woe)
# we plot again excluding these states

plot_by_woe(df_temp.iloc[2:-2, :])

# now NV has the lowest woe but also has low observations so we combine it with th next lowest woe state that has substantial observations
# so now we combinie NE IA NV FL as one, NV has no data so add it to highest risk state(assuming the worst risk for NV) that is FL, also we notice the next 3 states have similar woe so we combine them as well in this group
# combine NE, IA, ND, NV, FL, HI, AL
# also the last 5 states can be combined as low observations
# combine WV, NH, WY, DC, ME, ID
# now plot the remaining states
plot_by_woe(df_temp.iloc[6:-6, :])

# mow we create categogies based on woe and based on number of observation
# at the bottom of the graph till CA, observe NY and CA has enough obs and deserve a category of its own others can be combined
# combine [NM VA] [NY] [OK TM MO LA MD NM] [CA] [UT KY AZ NU] [AR MI PA CH MN] [RI MA DE SD IN] [GA WA OR] [WI MT(both are apart in woe but low obs)] [TX] [IL CT] [KS SC CO VT AK MS(combined because low obs)]
# previous combine NE, IA, ND, NV, FL, HI, AL -  refrence dummy
# combine WV, NH, WY, DC, ME, ID

# now we create the dummy variables and keep one as a reference variable
df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'], df_inputs_prepr['addr_state:IA'],
                                                         df_inputs_prepr['addr_state:NV'], df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'], df_inputs_prepr['addr_state:AL']])

# f_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL']

df_inputs_prepr['addr_state:NM_VA'] = sum(
    [df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])
df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                                      df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'], df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])
df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum(
    [df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'], df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])
df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                                   df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'], df_inputs_prepr['addr_state:MN']])
df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                                   df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'], df_inputs_prepr['addr_state:IN']])
df_inputs_prepr['addr_state:GA_WA_OR'] = sum(
    [df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'], df_inputs_prepr['addr_state:OR']])
df_inputs_prepr['addr_state:WI_MT'] = sum(
    [df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])
df_inputs_prepr['addr_state:IL_CT'] = sum(
    [df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])
df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                                      df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'], df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])
df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                                      df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'], df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])

# df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID']

loan_data['verification_status'].unique()

df_temp = woe_discrete(
    df_inputs_prepr, 'verification_status', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# loan_data.columns.values

loan_data['purpose'].unique()

df_temp = woe_discrete(df_inputs_prepr, 'purpose', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# We combine 'educational', 'small_business', 'wedding', 'renewable_energy', 'moving', 'house' in one category: 'educ__sm_b__wedd__ren_en__mov__house'.
# We combine 'other', 'medical', 'vacation' in one category: 'oth__med__vacation'.
# We combine 'major_purchase', 'car', 'home_improvement' in one category: 'major_purch__car__home_impr'.
# We leave 'debt_consolidtion' in a separate category.
# We leave 'credit_card' in a separate category.
# 'educ__sm_b__wedd__ren_en__mov__house' will be the reference category.
df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                       df_inputs_prepr['purpose:wedding'], df_inputs_prepr[
                                                                           'purpose:renewable_energy'],
                                                                       df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                                     df_inputs_prepr['purpose:vacation']])
df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                              df_inputs_prepr['purpose:home_improvement']])


# 'initial_list_status'
df_temp = woe_discrete(
    df_inputs_prepr, 'initial_list_status', df_targets_prepr)
df_temp

plot_by_woe(df_temp)
# We plot the weight of evidence values.

# now we will process the continuous variables
# wew can use the function created earlier but these a differecce
# in earlier catogorical data say state, one state is very different from other so we can order them as their WoE, however in continuous variable we will order them as per their value and not WoE


def woe_ordered_continuos(df, discrete_variable_name, good_bad_variable_df):
    temp1 = df[discrete_variable_name]
    temp2 = good_bad_variable_df
    # combines independent and dependednt variable
    df = pd.concat([temp1, temp2], axis=1)
    count_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].count()
    mean_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].mean()
    df = pd.concat([count_df1, mean_df1], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    # adding column, number of goods
    df['n_good'] = df['prop_good'] * df['n_obs']
    # adding column, number of bads
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    # df = df.sort_values(['WoE'])          #commented out these two lines of code
    # df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# no changed in plot_by_woe()


df_inputs_prepr['term_int'].unique()

df_temp = woe_ordered_continuos(df_inputs_prepr, 'term_int', df_targets_prepr)
df_temp

df_inputs_prepr['term_36'] = np.where(df_inputs_prepr['term_int'] == 36, 1, 0)
df_inputs_prepr['term_60'] = np.where(df_inputs_prepr['term_int'] == 60, 1, 0)

df_inputs_prepr['emp_length_int'].unique()

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'emp_length_int', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

df_inputs_prepr['emp_length:0'] = np.where(
    df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
df_inputs_prepr['emp_length:1'] = np.where(
    df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
df_inputs_prepr['emp_length:2-4'] = np.where(
    df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prepr['emp_length:5-6'] = np.where(
    df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prepr['emp_length:7-9'] = np.where(
    df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
df_inputs_prepr['emp_length:10'] = np.where(
    df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)


df_inputs_prepr['mths_since_issue_d'].unique()

# fine classing of mths_since_issue_d
# pd.cut(series, number of categories)- creates a new series which is a categorical variable with as many number of categories as specified, indicating the interval where each observation from the series lies
# creating new column(mths_since_issue_d_factor) with these 50 categories of (mths_since_issue_d)

df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(
    df_inputs_prepr['mths_since_issue_d'], 50)


# mths_since_issue_d_factor will have values like 43.2, 45.0 this is the range in where the value of mths_since_issue_d lies hence categoriesed
df_inputs_prepr['mths_since_issue_d_factor']

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'mths_since_issue_d_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)  # 90 is the rotation of labels in x axis

# now we check number of obs in each category and woe to combine them - coarse classing
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(38)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<38-39'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<40-41'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<42-48'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<49-52'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<53-64'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<65-84'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(
    range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)

# now we handle interest rates- first fine classing

df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)

df_inputs_prepr['int_rate_factor'].unique()

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'int_rate_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# coarse classing int rate, here the values are not integers so we will have to give the range, (so not we will not use the isin fn of np.where)

df_inputs_prepr['int_rate:<9.548'] = np.where(
    df_inputs_prepr['int_rate'] < 9.548, 1, 0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where(
    (df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] < 12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where(
    (df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] < 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where(
    (df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] < 20.281), 1, 0)
df_inputs_prepr['int_rate:>20.281'] = np.where(
    df_inputs_prepr['int_rate'] > 20.281, 1, 0)

df_inputs_prepr['funded_amnt_factor'] = pd.cut(
    df_inputs_prepr['funded_amnt'], 50)

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'funded_amnt_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# since the funded amount variable has very different woe and is kind of around a mean horizontal value, there no point coarse classing it or using it in our PD model

# mths_since_earliest_cr_line
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(
    df_inputs_prepr['mths_since_earliest_cr_line'], 50)
# fine classing
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

plot_by_woe(df_temp.iloc[6:, :], 90)
# We plot the weight of evidence values.

# We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(
    range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)

df_inputs_prepr['delinq_2yrs'].unique()

# delinq_2yrs
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'delinq_2yrs', df_targets_prepr)
# We calculate weight of evidence.
df_temp

plot_by_woe(df_temp)

# Categories: 0, 1-3, >=4
df_inputs_prepr['delinq_2yrs:0'] = np.where(
    (df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
df_inputs_prepr['delinq_2yrs:1-3'] = np.where(
    (df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
df_inputs_prepr['delinq_2yrs:>=4'] = np.where(
    (df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)

df_inputs_prepr['delinq_2yrs:>=4'].head()


# inq_last_6mths
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'inq_last_6mths', df_targets_prepr)

df_temp

plot_by_woe(df_temp)

# Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_prepr['inq_last_6mths:0'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
df_inputs_prepr['inq_last_6mths:1-2'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:3-6'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)

df_inputs_prepr['inq_last_6mths:>6'].head()


# open_acc
df_temp = woe_ordered_continuos(df_inputs_prepr, 'open_acc', df_targets_prepr)

df_temp

plot_by_woe(df_temp, 90)

plot_by_woe(df_temp.iloc[: 40, :], 90)

# Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_prepr['open_acc:0'] = np.where(
    (df_inputs_prepr['open_acc'] == 0), 1, 0)
df_inputs_prepr['open_acc:1-3'] = np.where(
    (df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
df_inputs_prepr['open_acc:4-12'] = np.where(
    (df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
df_inputs_prepr['open_acc:13-17'] = np.where(
    (df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
df_inputs_prepr['open_acc:18-22'] = np.where(
    (df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
df_inputs_prepr['open_acc:23-25'] = np.where(
    (df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
df_inputs_prepr['open_acc:26-30'] = np.where(
    (df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
df_inputs_prepr['open_acc:>=31'] = np.where(
    (df_inputs_prepr['open_acc'] >= 31), 1, 0)

# pub_rec
df_temp = woe_ordered_continuos(df_inputs_prepr, 'pub_rec', df_targets_prepr)

df_temp

plot_by_woe(df_temp, 90)

# Categories '0-2', '3-4', '>=5'
df_inputs_prepr['pub_rec:0-2'] = np.where(
    (df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
df_inputs_prepr['pub_rec:3-4'] = np.where(
    (df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
df_inputs_prepr['pub_rec:>=5'] = np.where(
    (df_inputs_prepr['pub_rec'] >= 5), 1, 0)

# total_acc
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'total_acc_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = np.where(
    (df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = np.where(
    (df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = np.where(
    (df_inputs_prepr['total_acc'] >= 52), 1, 0)

df_inputs_prepr['total_acc:>=52'].head()

# acc_now_delinq
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'acc_now_delinq', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# Categories: '0', '>=1'
df_inputs_prepr['acc_now_delinq:0'] = np.where(
    (df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
df_inputs_prepr['acc_now_delinq:>=1'] = np.where(
    (df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)

# total_rev_hi_lim
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(
    df_inputs_prepr['total_rev_hi_lim'], 2000)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'total_rev_hi_lim_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp.iloc[: 50, :], 90)

# Categories
# '<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)

# installment
df_inputs_prepr['installment_factor'] = pd.cut(
    df_inputs_prepr['installment'], 50)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'installment_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# now we will process annual income- this is an important one

df_inputs_prepr['annual_inc_factor'] = pd.cut(
    df_inputs_prepr['annual_inc'], 50)
# fine classing
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp

# when we cut annual income into 50 categories we find that 94% of the data is in first category(from prop_n_obs)
# so we need to cut it in more categories, well cut it in 100 categories

df_inputs_prepr['annual_inc_factor'] = pd.cut(
    df_inputs_prepr['annual_inc'], 100)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp

# still we find that 62% obs are in first class, we need to segregate the [-5243.882, 73294.82] income class and treat it diferently, also very few obs in high income class
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
# we create a temp var with annual income < 14000, we will make classes for this first
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, :]


df_inputs_prepr_temp.head()

df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(
    df_inputs_prepr_temp['annual_inc'], 50)

df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'annual_inc_factor', df_targets_prepr[df_inputs_prepr_temp.index])

df_temp

plot_by_woe(df_temp, 90)

# we observe that number of obs are low in first few classes and then woe is similar in next few so we combine till <20000.
# WoE is monotonically increasing with income after 20000, so we split income in 10 equal categories, each with width of 10k.
# because of fewer obs in >140000 we combine it into one class
df_inputs_prepr['annual_inc:<20K'] = np.where(
    (df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 140000), 1, 0)

# mths_since_last_delinq
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(
    df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(
    df_inputs_prepr_temp['mths_since_last_delinq'], 50)
df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_targets_prepr[df_inputs_prepr_temp.index])

df_temp

plot_by_woe(df_temp, 90)

# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)

# dti
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)
# Here we do fine-classing: using the 'cut' method
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'dti_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, :]

df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)
#  fine-classing
df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'dti_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp

plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.

# Categories:
df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
df_inputs_prepr['dti:1.4-3.5'] = np.where(
    (df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
df_inputs_prepr['dti:3.5-7.7'] = np.where(
    (df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
df_inputs_prepr['dti:7.7-10.5'] = np.where(
    (df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
df_inputs_prepr['dti:10.5-16.1'] = np.where(
    (df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
df_inputs_prepr['dti:16.1-20.3'] = np.where(
    (df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
df_inputs_prepr['dti:20.3-21.7'] = np.where(
    (df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
df_inputs_prepr['dti:21.7-22.4'] = np.where(
    (df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
df_inputs_prepr['dti:22.4-35'] = np.where(
    (df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)

# mths_since_last_record
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(
    df_inputs_prepr['mths_since_last_record'])]
# sum(loan_data_temp['mths_since_last_record'].isnull())
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(
    df_inputs_prepr_temp['mths_since_last_record'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'mths_since_last_record_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp

plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.

# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where(
    (df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)

# making all above changes in test data as well- no need to asses woe
# we first update the loan_data_inputs_train with all the preprocessind done (dummy variables created for train )
loan_data_inputs_train = df_inputs_prepr


loan_data_inputs_train.columns.values

# then we make all these changes to test data using the same code we used for train data
# we can run the whole code from where we made prepr df, i am copying the code again, we dont need to see the woe analysis, also some of the comments may be copied and not make sense, can be ignored

# Here we are creating a different df for preprocessing the data, after pre processing the data will be returned to loan_data_inputs_train, loan_data_targets_train
# these preprocessing steps will also be applied to test data using the same following code at that time we just need to replace loan_data_inputs_train, loan_data_targets_train with loan_data_inputs_test, loan_data_targets_test


# code is copied from here, from above changes made in training data also made to test
# df_inputs_prepr = loan_data_inputs_train
# df_targets_prepr = loan_data_targets_train

df_inputs_prepr = loan_data_inputs_test
df_targets_prepr = loan_data_targets_test

df_inputs_prepr['grade'].unique()
# we will create a dataframe containig only grades from inputs and good_bad from the target
df1 = pd.concat([df_inputs_prepr['grade'], df_targets_prepr], axis=1)
df1.head()

# now we will cal weight of evidence for grade
# groupby, we are grouping the df1 based on first column(grades) all similar grade A together, (group values(A,B) become indexes), then we also want it show the number of No observations (good and bad)in a group
# parameter of group by- DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True)
# asindex = false doesnot let group values become indexes

count_df = df1.groupby(df1.columns.values[0], as_index=False)[
    df1.columns.values[1]].count()

# we want % of good and bad for each grade to cal WOE, since good is 1 and bad is 0, we can cal % of good, and bad = 1-%good
# we can cal the good % by using mean() this will give the % of good borrowers in each grade. since there are only good and bad in second column
# mean hear in this particular case means Good(1s)+bad(0s) in a particlar grade/total values in that grade
mean_df = df1.groupby(df1.columns.values[0], as_index=False)[
    df1.columns.values[1]].mean()
df1 = pd.concat([count_df, mean_df], axis=1)
df1

df1 = df1.iloc[:, [0, 1, 3]]
print(df1.columns)

# renaming column
df1.columns = [df1.columns.values[0], 'n_obs', 'prop_good']
df1

# proportion of observation in each row
df1['prop_n_obs'] = df1['n_obs'] / df1['n_obs'].sum()
df1

df1['prop_n_obs'].sum()

df1['prop_good'].sum()

# so far df1 - n_obs(number of obs in a grade), prop_good(% of goods in a grade), prop_n_obs (No of goods in a grade/all goods in all grades)

# adding column, number of goods
df1['n_good'] = df1['prop_good'] * df1['n_obs']
# adding column, number of bads
df1['n_bad'] = (1 - df1['prop_good']) * df1['n_obs']
df1

# proportion of good borrowers = good in a category/good in all categories

df1['prop_n_good'] = df1['n_good'] / df1['n_good'].sum()
df1['prop_n_bad'] = df1['n_bad'] / df1['n_bad'].sum()
df1

# weight of evidence(for variable grade) = ln(% of good / % of bad)

df1['WoE'] = np.log(df1['prop_n_good'] / df1['prop_n_bad'])
df1

# sorting by WoE, higher up means prob of default is high

df1 = df1.sort_values(['WoE'])
df1 = df1.reset_index(drop=True)
df1

# we are trying to see how much difference is there in prop_good and WOE in adjacent rows. .diff() substracts the value in row with value in row above-The first value in the resulting Series is NaN because there is no previous value to subtract for the first row.
# The subsequent values represent the differences between consecutive rows.

df1['diff_prop_good'] = df1['prop_good'].diff().abs()
df1['diff_WoE'] = df1['WoE'].diff().abs()
df1

# now we cal the information value, IV = sum across the whole column(%good - %bad)*WoE
df1['IV'] = (df1['prop_n_good'] - df1['prop_n_bad']) * df1['WoE']
df1

# summing across the column to get IV for variable grades

df1['IV'] = df1['IV'].sum()
df1

# automating the IV calculation by defining a function that takes (training input dataframe, variable name, training output df)


def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    temp1 = df[discrete_variable_name]
    temp2 = good_bad_variable_df
    # combines independent and dependednt variable
    df = pd.concat([temp1, temp2], axis=1)
    count_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].count()
    mean_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].mean()
    df = pd.concat([count_df1, mean_df1], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    # adding column, number of goods
    df['n_good'] = df['prop_good'] * df['n_obs']
    # adding column, number of bads
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# crosschecking fn on grade variable
df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp

# importing for visulization
sns.set()

# creating a function to plot the woe, we will pass the df with WOE number, also some variable categories(a,b,c,d in grade)
# can be long so we want to rotate the variable name such that it is legible(by default value is 0, only to be passed if rotation required)


def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    # labels/categories of variable are listed in first column, also converting them to string, also converting it to np array as matplotlib performs better with np array
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weights of Evidence')
    plt.title(str('Weight of Evidence by' + df_WoE.columns[0]))
    plt.xticks(rotation=rotation_of_x_axis_labels)


plot_by_woe(df_temp)

# created a list of dummy variable in excel file loan_data_2007-2014, we have kept the category with worst WOE as a reference category, g in grade for example

# using functions on other variables
df_temp = woe_discrete(df_inputs_prepr, 'home_ownership', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# we notice that the category other, none, any have very few observation, so we combine them with the riskiets category rent
# while setting up dummy variable we want to keep categories with similat WoE together so 3 categories here 1 other+none+any+rent 2 own 3 mortgage
# below dummy var is 1 if any of the options is 1 hence all of them are combined
df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum(
    [df_inputs_prepr['home_ownership:RENT'], df_inputs_prepr['home_ownership:NONE'], df_inputs_prepr['home_ownership:OTHER'], df_inputs_prepr['home_ownership:ANY'],])

df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY']

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


df_inputs_prepr['addr_state'].unique()
df_inputs_prepr['addr_state'].nunique()

# there are 50 unique address states- well have to combine them at somepoint to reduce number of dummy variables

df_temp = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr)
df_temp
plot_by_woe(df_temp)

# northdacota not represented as no obs, setting a dummy just in case

if ['addr_state:ND'] in df_inputs_prepr.columns.values:
    pass
else:
    df_inputs_prepr['addr_state:ND'] = 0

# df_inputs_prepr['addr_state:ND']

# from the data of df_temp = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr) we observe that
# the first two states NE IA and last two states ME ID have very few observation so we add them to the 3rd and 3rd last states (worst and best based on similarities in their woe)
# we plot again excluding these states

plot_by_woe(df_temp.iloc[2:-2, :])

# now NV has the lowest woe but also has low observations so we combine it with th next lowest woe state that has substantial observations
# so now we combinie NE IA NV FL as one, NV has no data so add it to highest risk state(assuming the worst risk for NV) that is FL, also we notice the next 3 states have similar woe so we combine them as well in this group
# combine NE, IA, ND, NV, FL, HI, AL
# also the last 5 states can be combined as low observations
# combine WV, NH, WY, DC, ME, ID
# now plot the remaining states
plot_by_woe(df_temp.iloc[6:-6, :])

# mow we create categogies based on woe and based on number of observation
# at the bottom of the graph till CA, observe NY and CA has enough obs and deserve a category of its own others can be combined
# combine [NM VA] [NY] [OK TM MO LA MD NM] [CA] [UT KY AZ NU] [AR MI PA CH MN] [RI MA DE SD IN] [GA WA OR] [WI MT(both are apart in woe but low obs)] [TX] [IL CT] [KS SC CO VT AK MS(combined because low obs)]
# previous combine NE, IA, ND, NV, FL, HI, AL -  refrence dummy
# combine WV, NH, WY, DC, ME, ID

# now we create the dummy variables and keep one as a reference variable
df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'], df_inputs_prepr['addr_state:IA'],
                                                         df_inputs_prepr['addr_state:NV'], df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'], df_inputs_prepr['addr_state:AL']])

# f_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL']

df_inputs_prepr['addr_state:NM_VA'] = sum(
    [df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])
df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                                      df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'], df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])
df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum(
    [df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'], df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])
df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                                   df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'], df_inputs_prepr['addr_state:MN']])
df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                                   df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'], df_inputs_prepr['addr_state:IN']])
df_inputs_prepr['addr_state:GA_WA_OR'] = sum(
    [df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'], df_inputs_prepr['addr_state:OR']])
df_inputs_prepr['addr_state:WI_MT'] = sum(
    [df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])
df_inputs_prepr['addr_state:IL_CT'] = sum(
    [df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])
df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                                      df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'], df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])
df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                                      df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'], df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])

# df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID']

loan_data['verification_status'].unique()

df_temp = woe_discrete(
    df_inputs_prepr, 'verification_status', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# loan_data.columns.values

loan_data['purpose'].unique()

df_temp = woe_discrete(df_inputs_prepr, 'purpose', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# We combine 'educational', 'small_business', 'wedding', 'renewable_energy', 'moving', 'house' in one category: 'educ__sm_b__wedd__ren_en__mov__house'.
# We combine 'other', 'medical', 'vacation' in one category: 'oth__med__vacation'.
# We combine 'major_purchase', 'car', 'home_improvement' in one category: 'major_purch__car__home_impr'.
# We leave 'debt_consolidtion' in a separate category.
# We leave 'credit_card' in a separate category.
# 'educ__sm_b__wedd__ren_en__mov__house' will be the reference category.
df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                       df_inputs_prepr['purpose:wedding'], df_inputs_prepr[
                                                                           'purpose:renewable_energy'],
                                                                       df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                                     df_inputs_prepr['purpose:vacation']])
df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                              df_inputs_prepr['purpose:home_improvement']])


# 'initial_list_status'
df_temp = woe_discrete(
    df_inputs_prepr, 'initial_list_status', df_targets_prepr)
df_temp

plot_by_woe(df_temp)
# We plot the weight of evidence values.

# now we will process the continuous variables
# wew can use the function created earlier but these a differecce
# in earlier catogorical data say state, one state is very different from other so we can order them as their WoE, however in continuous variable we will order them as per their value and not WoE


def woe_ordered_continuos(df, discrete_variable_name, good_bad_variable_df):
    temp1 = df[discrete_variable_name]
    temp2 = good_bad_variable_df
    # combines independent and dependednt variable
    df = pd.concat([temp1, temp2], axis=1)
    count_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].count()
    mean_df1 = df.groupby(df.columns.values[0], as_index=False)[
        df.columns.values[1]].mean()
    df = pd.concat([count_df1, mean_df1], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    # adding column, number of goods
    df['n_good'] = df['prop_good'] * df['n_obs']
    # adding column, number of bads
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    # df = df.sort_values(['WoE'])          #commented out these two lines of code
    # df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# no changed in plot_by_woe()


df_inputs_prepr['term_int'].unique()

df_temp = woe_ordered_continuos(df_inputs_prepr, 'term_int', df_targets_prepr)
df_temp

df_inputs_prepr['term_36'] = np.where(df_inputs_prepr['term_int'] == 36, 1, 0)
df_inputs_prepr['term_60'] = np.where(df_inputs_prepr['term_int'] == 60, 1, 0)

df_inputs_prepr['emp_length_int'].unique()

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'emp_length_int', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

df_inputs_prepr['emp_length:0'] = np.where(
    df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
df_inputs_prepr['emp_length:1'] = np.where(
    df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
df_inputs_prepr['emp_length:2-4'] = np.where(
    df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prepr['emp_length:5-6'] = np.where(
    df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prepr['emp_length:7-9'] = np.where(
    df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
df_inputs_prepr['emp_length:10'] = np.where(
    df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)


df_inputs_prepr['mths_since_issue_d'].unique()

# fine classing of mths_since_issue_d
# pd.cut(series, number of categories)- creates a new series which is a categorical variable with as many number of categories as specified, indicating the interval where each observation from the series lies
# creating new column(mths_since_issue_d_factor) with these 50 categories of (mths_since_issue_d)

df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(
    df_inputs_prepr['mths_since_issue_d'], 50)


# mths_since_issue_d_factor will have values like 43.2, 45.0 this is the range in where the value of mths_since_issue_d lies hence categoriesed
df_inputs_prepr['mths_since_issue_d_factor']

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'mths_since_issue_d_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)  # 90 is the rotation of labels in x axis

# now we check number of obs in each category and woe to combine them - coarse classing
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(38)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<38-39'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<40-41'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<42-48'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<49-52'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<53-64'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
df_inputs_prepr['mths_since_issue_d:<65-84'] = np.where(
    df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(
    range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)

# now we handle interest rates- first fine classing

df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)

df_inputs_prepr['int_rate_factor'].unique()

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'int_rate_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# coarse classing int rate, here the values are not integers so we will have to give the range, (so not we will not use the isin fn of np.where)

df_inputs_prepr['int_rate:<9.548'] = np.where(
    df_inputs_prepr['int_rate'] < 9.548, 1, 0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where(
    (df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] < 12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where(
    (df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] < 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where(
    (df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] < 20.281), 1, 0)
df_inputs_prepr['int_rate:>20.281'] = np.where(
    df_inputs_prepr['int_rate'] > 20.281, 1, 0)

df_inputs_prepr['funded_amnt_factor'] = pd.cut(
    df_inputs_prepr['funded_amnt'], 50)

df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'funded_amnt_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# since the funded amount variable has very different woe and is kind of around a mean horizontal value, there no point coarse classing it or using it in our PD model

# mths_since_earliest_cr_line
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(
    df_inputs_prepr['mths_since_earliest_cr_line'], 50)
# fine classing
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

plot_by_woe(df_temp.iloc[6:, :], 90)
# We plot the weight of evidence values.

# We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(
    df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(
    range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)

df_inputs_prepr['delinq_2yrs'].unique()

# delinq_2yrs
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'delinq_2yrs', df_targets_prepr)
# We calculate weight of evidence.
df_temp

plot_by_woe(df_temp)

# Categories: 0, 1-3, >=4
df_inputs_prepr['delinq_2yrs:0'] = np.where(
    (df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
df_inputs_prepr['delinq_2yrs:1-3'] = np.where(
    (df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
df_inputs_prepr['delinq_2yrs:>=4'] = np.where(
    (df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)

df_inputs_prepr['delinq_2yrs:>=4'].head()


# inq_last_6mths
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'inq_last_6mths', df_targets_prepr)

df_temp

plot_by_woe(df_temp)

# Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_prepr['inq_last_6mths:0'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
df_inputs_prepr['inq_last_6mths:1-2'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:3-6'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6'] = np.where(
    (df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)

df_inputs_prepr['inq_last_6mths:>6'].head()


# open_acc
df_temp = woe_ordered_continuos(df_inputs_prepr, 'open_acc', df_targets_prepr)

df_temp

plot_by_woe(df_temp, 90)

plot_by_woe(df_temp.iloc[: 40, :], 90)

# Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_prepr['open_acc:0'] = np.where(
    (df_inputs_prepr['open_acc'] == 0), 1, 0)
df_inputs_prepr['open_acc:1-3'] = np.where(
    (df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
df_inputs_prepr['open_acc:4-12'] = np.where(
    (df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
df_inputs_prepr['open_acc:13-17'] = np.where(
    (df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
df_inputs_prepr['open_acc:18-22'] = np.where(
    (df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
df_inputs_prepr['open_acc:23-25'] = np.where(
    (df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
df_inputs_prepr['open_acc:26-30'] = np.where(
    (df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
df_inputs_prepr['open_acc:>=31'] = np.where(
    (df_inputs_prepr['open_acc'] >= 31), 1, 0)

# pub_rec
df_temp = woe_ordered_continuos(df_inputs_prepr, 'pub_rec', df_targets_prepr)

df_temp

plot_by_woe(df_temp, 90)

# Categories '0-2', '3-4', '>=5'
df_inputs_prepr['pub_rec:0-2'] = np.where(
    (df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
df_inputs_prepr['pub_rec:3-4'] = np.where(
    (df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
df_inputs_prepr['pub_rec:>=5'] = np.where(
    (df_inputs_prepr['pub_rec'] >= 5), 1, 0)

# total_acc
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'total_acc_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = np.where(
    (df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = np.where(
    (df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = np.where(
    (df_inputs_prepr['total_acc'] >= 52), 1, 0)

df_inputs_prepr['total_acc:>=52'].head()

# acc_now_delinq
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'acc_now_delinq', df_targets_prepr)
df_temp

plot_by_woe(df_temp)

# Categories: '0', '>=1'
df_inputs_prepr['acc_now_delinq:0'] = np.where(
    (df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
df_inputs_prepr['acc_now_delinq:>=1'] = np.where(
    (df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)

# total_rev_hi_lim
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(
    df_inputs_prepr['total_rev_hi_lim'], 2000)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'total_rev_hi_lim_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp.iloc[: 50, :], 90)

# Categories
# '<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where(
    (df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)

# installment
df_inputs_prepr['installment_factor'] = pd.cut(
    df_inputs_prepr['installment'], 50)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'installment_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# now we will process annual income- this is an important one

df_inputs_prepr['annual_inc_factor'] = pd.cut(
    df_inputs_prepr['annual_inc'], 50)
# fine classing
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp

# when we cut annual income into 50 categories we find that 94% of the data is in first category(from prop_n_obs)
# so we need to cut it in more categories, well cut it in 100 categories

df_inputs_prepr['annual_inc_factor'] = pd.cut(
    df_inputs_prepr['annual_inc'], 100)
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp

# still we find that 62% obs are in first class, we need to segregate the [-5243.882, 73294.82] income class and treat it diferently, also very few obs in high income class
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
# we create a temp var with annual income < 14000, we will make classes for this first
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, :]


df_inputs_prepr_temp.head()

df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(
    df_inputs_prepr_temp['annual_inc'], 50)

df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'annual_inc_factor', df_targets_prepr[df_inputs_prepr_temp.index])

df_temp

plot_by_woe(df_temp, 90)

# we observe that number of obs are low in first few classes and then woe is similar in next few so we combine till <20000.
# WoE is monotonically increasing with income after 20000, so we split income in 10 equal categories, each with width of 10k.
# because of fewer obs in >140000 we combine it into one class
df_inputs_prepr['annual_inc:<20K'] = np.where(
    (df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where(
    (df_inputs_prepr['annual_inc'] > 140000), 1, 0)

# mths_since_last_delinq
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(
    df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(
    df_inputs_prepr_temp['mths_since_last_delinq'], 50)
df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_targets_prepr[df_inputs_prepr_temp.index])

df_temp

plot_by_woe(df_temp, 90)

# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where(
    (df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)

# dti
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)
# Here we do fine-classing: using the 'cut' method
df_temp = woe_ordered_continuos(
    df_inputs_prepr, 'dti_factor', df_targets_prepr)
df_temp

plot_by_woe(df_temp, 90)

# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, :]

df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)
#  fine-classing
df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'dti_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp

plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.

# Categories:
df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
df_inputs_prepr['dti:1.4-3.5'] = np.where(
    (df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
df_inputs_prepr['dti:3.5-7.7'] = np.where(
    (df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
df_inputs_prepr['dti:7.7-10.5'] = np.where(
    (df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
df_inputs_prepr['dti:10.5-16.1'] = np.where(
    (df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
df_inputs_prepr['dti:16.1-20.3'] = np.where(
    (df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
df_inputs_prepr['dti:20.3-21.7'] = np.where(
    (df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
df_inputs_prepr['dti:21.7-22.4'] = np.where(
    (df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
df_inputs_prepr['dti:22.4-35'] = np.where(
    (df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)

# mths_since_last_record
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(
    df_inputs_prepr['mths_since_last_record'])]
# sum(loan_data_temp['mths_since_last_record'].isnull())
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(
    df_inputs_prepr_temp['mths_since_last_record'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuos(
    df_inputs_prepr_temp, 'mths_since_last_record_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp

plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.

# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where(
    (df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where(
    (df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)


# copied code ends changes made in training also applied to test data

# all similar changes made to test data
loan_data_inputs_test = df_inputs_prepr

loan_data_inputs_test.shape
loan_data_inputs_train.shape

test_cols = set(loan_data_inputs_test.columns)
train_cols = set(loan_data_inputs_train.columns)

print("Columns in test but not in train:", test_cols - train_cols)
print("Columns in train but not in test:", train_cols - test_cols)

# saving all the dfs as csv

loan_data_inputs_test.to_csv('loan_data_inputs_test.csv', index=True)
loan_data_inputs_train.to_csv('loan_data_inputs_train.csv', index=True)
loan_data_targets_train.to_csv('loan_data_targets_train.csv', index=True)
loan_data_targets_test.to_csv('loan_data_targets_test.csv', index=True)

# PD model starts- Logistis reg

# now we start with the PD modelling
# we will use logistic regression
# our defualt definition was: classify as default if loan status: Charged Off, Default, Does not meet the credit policy. Status:Charged Off, Late (31-120 days)

loan_data_targets_train.head()

loan_data_inputs_train.head()
