# --final monthly- edited attempt 2

# importing libraries

from scipy.stats import chi2
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats.mstats import winsorize
from statsmodels.stats.sandwich_covariance import cov_hac
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# function to remove months with less trading data


def fn_monthoutlier(dailyreturns_data):
    for share_1 in dailyreturns_data.columns:
        tr_days_month = dailyreturns_data[share_1].resample('ME').count()
        less_tr_mon = tr_days_month[tr_days_month < 10].dropna().index
        dailyreturns_data.loc[dailyreturns_data.index.to_period('M').isin(
            less_tr_mon.to_period('M')), share_1] = np.nan
    return dailyreturns_data


# Semibeta calculation- calculates monthly semibeta
def fn_semibeta_cal(signed_stock_returns, signed_market_returns, all_market_returns):
    numerator_df = signed_stock_returns * signed_market_returns.values
    denomerator_df = all_market_returns ** 2
    sum_numerator_df = numerator_df.resample('ME').sum(min_count=1)
    sum_denomerator_df = denomerator_df.resample('ME').sum(min_count=1)
    motnhly_semib_df = sum_numerator_df / sum_denomerator_df.values
    return motnhly_semib_df


def fn_mixed_semib(signed_stock_returns, signed_market_returns, all_market_returns):
    numerator_df = signed_stock_returns * signed_market_returns.values
    denomerator_df = all_market_returns ** 2
    sum_numerator_df = numerator_df.resample('ME').sum(min_count=1)
    sum_denomerator_df = denomerator_df.resample('ME').sum(min_count=1)
    motnhly_semib_df = -sum_numerator_df / sum_denomerator_df.values  # sign change
    return motnhly_semib_df


def semibeta_up(r_all, signed_market_returns):
    numerator_df = r_all * signed_market_returns.values
    denomerator_df = signed_market_returns ** 2
    sum_numerator_df = numerator_df.resample('ME').sum(min_count=1)
    sum_denomerator_df = denomerator_df.resample('ME').sum(min_count=1)
    beta_up = sum_numerator_df / sum_denomerator_df.values
    return beta_up


def semibeta_down(r_all, signed_market_returns):
    numerator_df = r_all * signed_market_returns.values
    denomerator_df = signed_market_returns ** 2
    sum_numerator_df = numerator_df.resample('ME').sum(min_count=1)
    sum_denomerator_df = denomerator_df.resample('ME').sum(min_count=1)
    beta_up = sum_numerator_df / sum_denomerator_df.values
    return beta_up

# Function to winsorize column


def fn_win(Win_series, lower_limit, upper_limit):
    return pd.Series(winsorize(Win_series, limits=(lower_limit, upper_limit)), index=Win_series.index)


# function to calculate summary statistics
def summarystatistics_fn(motnhly_semib_df):
    crossectional_mean = motnhly_semib_df.mean(axis=1)
    crossectional_std = motnhly_semib_df.std(axis=1, ddof=1)
    crossectional_median = motnhly_semib_df.median(axis=1)
    timeseries_mean = crossectional_mean.mean()
    ts_std = crossectional_std.mean()
    ts_median = crossectional_median.mean()
    return timeseries_mean, ts_median, ts_std


def averagecross_correlation(matrix1, matrix2):
    correlations = []
    for i in range(len(matrix1)):
        corr = matrix1.iloc[i].corr(matrix2.iloc[i], 'pearson')
        correlations.append(corr)
    return np.mean(correlations)

# neweywest functions for SE and tstat


def fn_newey_se(coeff, nlags):
    res_1 = sm.OLS(coeff, np.ones(len(coeff))).fit()
    neweycov = cov_hac(res_1, nlags=nlags)
    neweyse = np.sqrt(np.diag(neweycov))
    return neweyse


def fn_newey_tstat(coeff, mean_coeff, nlags):
    neweyse = fn_newey_se(coeff, nlags)
    nw_tstat = mean_coeff / neweyse
    p_values = 2 * (1 - norm.cdf(np.abs(nw_tstat)))
    return nw_tstat, p_values


def calculate_coskewness_cokurtosis(r, m):
    r_monthly_mean = r.resample('M').mean()
    m_monthly_mean = m.resample('M').mean()
    r_diff = r.groupby(r.index.to_period(
        'M')).transform(lambda x: x - x.mean())
    m_diff = m.groupby(m.index.to_period(
        'M')).transform(lambda x: x - x.mean())
    coskewness_num = r_diff.mul(m_diff.values**2, axis=0).resample('M').sum()
    coskewness_denom_r = r_diff.pow(2).resample(
        'M').sum()  # r_diff^2 sum per month
    coskewness_denom_m = m_diff.pow(2).resample(
        'M').sum()  # m_diff^2 sum per month
    coskewness = coskewness_num / \
        (coskewness_denom_r * coskewness_denom_m.values +
         1e-8)  # Avoid division by zero
    cokurtosis_num = r_diff.mul(m_diff.values**3, axis=0).resample('M').sum()
    cokurtosis_denom_r = r_diff.pow(2).resample('M').sum()
    cokurtosis_denom_m = m_diff.pow(2).resample('M').sum()
    cokurtosis_denom = (cokurtosis_denom_r *
                        cokurtosis_denom_m.values).pow(3/2)
    cokurtosis = cokurtosis_num / \
        (cokurtosis_denom + 1e-8)  # Avoid division by zero
    return coskewness, cokurtosis


# =============================================================================
#  Cleaning data
# =============================================================================
file_path = r"C:\\Users\\mayan\\OneDrive - University of Birmingham\\Desktop\\dessertation\\data\\CRPSsmall.csv"
thedata = pd.read_csv(file_path)
thedata['date'] = pd.to_datetime(thedata['date'])

# Set 'date' as the index without removing the column
thedata.set_index('date', drop=False, inplace=True)
relevant_columns = ['PRC', 'VOL', 'SHROUT', 'DLSTCD']
thedata['PRC'] = thedata['PRC'].abs()  # Take absolute values of PRC
thedata[relevant_columns] = thedata[relevant_columns].apply(pd.to_numeric)

# Identify PERMNOs where PRC is less than 5 and drop them
permno_to_drop = thedata[thedata['PRC'] < 5]['PERMNO'].unique()
thedata = thedata[~thedata['PERMNO'].isin(permno_to_drop)]
thedata['DLSTCD'] = pd.to_numeric(thedata['DLSTCD'])
thedata = thedata[thedata['DLSTCD'].isna()]

# dropping shares with low trading volume
average_volume = thedata.groupby('PERMNO')['VOL'].mean()
lower_15_percentile = average_volume.quantile(0.15)
low_volume_share_1s = average_volume[average_volume <
                                     lower_15_percentile].index
thedata = thedata[~thedata['PERMNO'].isin(low_volume_share_1s)]
number_of_low_volume_share_1s = len(low_volume_share_1s)
unique_dates = thedata['date'].unique()

# Identify PERMNOs that either:
# 1. Don't have entries for all unique dates
# 2. Are missing PRC values for any date
# and dropping them
permno_to_drop = thedata.groupby('PERMNO').filter(
    lambda x: len(x['date'].unique()) < len(
        unique_dates) or x['PRC'].isna().any()
)['PERMNO'].unique()
number_of_permno_dropped = len(permno_to_drop)
print(f"Number of PERMNOs to be dropped: {number_of_permno_dropped}")
thedata = thedata[~thedata['PERMNO'].isin(permno_to_drop)]
remaining_permno_count = thedata['PERMNO'].nunique()

thedata.drop(columns=['date'], inplace=True)
thedata.reset_index(inplace=True)

# calculating log returns
thedata = thedata.sort_values(by=['PERMNO', 'date'])
thedata['RET'] = thedata.groupby('PERMNO')['PRC'].apply(
    lambda x: np.log(x / x.shift(1))
).reset_index(level=0, drop=True)

# Step 3: Remove rows where the return ('RET') is NaN (the first date for each share_1)
thedata = thedata[~thedata['RET'].isna()]

# Backfill 0 returns in the RET column
thedata['RET'] = thedata['RET'].replace(0, np.nan).bfill()

# Remove specified columns
columns_to_remove = ['DLSTCD', 'VOL']
thedata = thedata.drop(columns=columns_to_remove)

# preparing famafrench data
famafrenchdata = pd.read_csv(
    r"C:\\Users\\mayan\\OneDrive - University of Birmingham\\Desktop\\dessertation\\data\\fama fench data 1.csv")
famafrenchdata['date'] = pd.to_datetime(famafrenchdata['date'])
ff_factors = famafrenchdata[['date', 'smb', 'hml', 'umd']]
thedata['date'] = pd.to_datetime(thedata['date'])
# Merge Fama-French factors into 'thedata' on 'date'
thedata = pd.merge(thedata, ff_factors, on='date', how='left')


columns_to_drop = ['PERMCO', 'CUSIP', 'RCRDDT', 'SHRCD', 'vwretd', 'ewretd']
thedata = thedata.drop(columns=columns_to_drop)

# calculating daily market returns
thedata['market_cap'] = thedata['PRC'] * thedata['SHROUT']
daily_market_return = thedata.groupby('date').apply(
    lambda x: np.sum(x['RET'] * x['market_cap']) / np.sum(x['market_cap']))
thedata['market_return'] = thedata['date'].map(daily_market_return)


# Preparing stocks and market returns
thedata['date'] = pd.to_datetime(thedata['date'])
temp_df = thedata[['PERMNO', 'date', 'RET']]
r = temp_df.pivot_table(index='date', columns='PERMNO', values='RET')
m = thedata[['date', 'market_return']]
m = m.drop_duplicates(subset=['date'])
m.set_index('date', inplace=True)
m.columns = ['daily market ret']


r_monthly = r.resample('M').sum()


# =============================================================================
# calculating factors
# =============================================================================

# segregating returns into positive and negative
r_pos = r.mask(r < 0, 0)
r_neg = r.mask(r > 0, 0)
m_pos = m.mask(m < 0, 0)
m_neg = m.mask(m > 0, 0)

# Handle outlier months
r = fn_monthoutlier(r)
m = fn_monthoutlier(m)
r_pos = fn_monthoutlier(r_pos)
r_neg = fn_monthoutlier(r_neg)
m_pos = fn_monthoutlier(m_pos)
m_neg = fn_monthoutlier(m_neg)


# semibeta calculation
beta_CAPM = fn_semibeta_cal(r, m, m)
beta_N = fn_semibeta_cal(r_neg, m_neg, m)
beta_P = fn_semibeta_cal(r_pos, m_pos, m)
beta_M_pos = fn_mixed_semib(r_neg, m_pos, m)
beta_M_neg = fn_mixed_semib(r_pos, m_neg, m)

# Preparing Fama-French factors
ff_factors_monthly = thedata[['date', 'smb', 'hml', 'umd']]
ff_factors_monthly.set_index('date', inplace=True)
ff_factors_monthly = ff_factors_monthly.resample('M').mean()
ff_factors_monthly.index = ff_factors_monthly.index.to_period('M')

# Step 1: Calculate Upside and Downside Betas, Coskewness, and Cokurtosis
beta_up = semibeta_up(r, m_pos)
beta_down = semibeta_down(r, m_neg)
coskewness, cokurtosis = calculate_coskewness_cokurtosis(r, m)

# Winsorize at 1% and 99% levels
elements_to_winsorize = [beta_up, beta_down, coskewness,
                         cokurtosis, beta_CAPM, beta_N, beta_P, beta_M_pos, beta_M_neg]
for element in elements_to_winsorize:
    element.apply(fn_win, lower_limit=0.01,
                  upper_limit=0.01, axis=0)

# Convert all column names to strings and indices to PeriodIndex
dataframes_to_convert = [beta_up, beta_down, r_monthly, beta_CAPM, beta_N,
                         beta_P, beta_M_pos, beta_M_neg, coskewness, cokurtosis, ff_factors_monthly]
for df in dataframes_to_convert:
    df.columns = df.columns.astype(str)
    df.index = pd.PeriodIndex(df.index, freq='M')


# =============================================================================
# semibeta summary stats
# =============================================================================

# Semibeta summary statistics
sumstat_index = ['Mean', 'Med', 'Std']
sumstat_col = ['B', 'B_N', 'B_P', 'B_M+', 'B_M-']
sumstat = pd.DataFrame(index=sumstat_index, columns=sumstat_col)
sumstat['B'] = summarystatistics_fn(beta_CAPM)
sumstat['B_N'] = summarystatistics_fn(beta_N)
sumstat['B_P'] = summarystatistics_fn(beta_P)
sumstat['B_M+'] = summarystatistics_fn(beta_M_pos)
sumstat['B_M-'] = summarystatistics_fn(beta_M_neg)

# Correlation between semibetas
betas = [beta_CAPM, beta_N, beta_P, beta_M_pos, beta_M_neg]
corr_matrix_label = ['B', 'B_N', 'B_P', 'B_M+', 'B_M-']
corr_matrix = pd.DataFrame(index=corr_matrix_label, columns=corr_matrix_label)
for i in range(len(betas)):
    for j in range(i, len(betas)):
        if i == j:
            corr_matrix.iloc[i, j] = 1.0
        else:
            flat1 = betas[i].values.flatten()
            flat2 = betas[j].values.flatten()
            combined_df = pd.DataFrame({'flat1': flat1, 'flat2': flat2})
            clean_df = combined_df.dropna()
            corr = np.corrcoef(clean_df['flat1'], clean_df['flat2'])[0, 1]
            corr_matrix.iloc[j, i] = corr


print("Semibeta Summary Statistics")
print(sumstat.to_string())
print("\nSemibeta Correlation Matrix")
print(corr_matrix.to_string())


# =============================================================================
# regression 1 - CAPM
# =============================================================================

lambda_0_reg1, lambda_CAPM_reg1, rsquared_reg1 = [], [], []
models = []
regression_counts = {}

for share_1 in beta_CAPM.columns:
    beta_values = beta_CAPM[share_1]
    r_values = r_monthly[share_1]

    # Align the indices
    aligned_data = pd.concat([beta_values, r_values],
                             axis=1, keys=['CAPM', 'r']).dropna()

    # If there's not enough data after alignment, skip this share

    if aligned_data.empty or len(aligned_data) < 2:
        print(f"Skipping share_1 {share_1} due to insufficient data.")
        continue

    x = sm.add_constant(aligned_data[['CAPM']], has_constant='add')
    y = aligned_data['r']
    model_reg1 = sm.OLS(y, x).fit()

    # Store the regression res_1
    models.append(model_reg1)
    lambda_0_reg1.append(model_reg1.params.get('const', np.nan))
    lambda_CAPM_reg1.append(model_reg1.params.get('CAPM', np.nan))
    rsquared_reg1.append(model_reg1.rsquared)
    regression_counts[share_1] = len(aligned_data)

coeff_reg1 = pd.DataFrame(
    {'lambda_0': lambda_0_reg1, 'lambda_CAPM': lambda_CAPM_reg1})

mean_coeff_reg1 = coeff_reg1.mean()

lambda_0_coef_reg1 = mean_coeff_reg1['lambda_0'] * 100
lambda_CAPM_coef_reg1 = mean_coeff_reg1['lambda_CAPM'] * 100

# Calculate Newey-West t-statistics
nlags = round(0.75 * len(r_monthly) ** (1/3))
nw_tstat0, p_values0 = fn_newey_tstat(
    coeff_reg1['lambda_0'], mean_coeff_reg1['lambda_0'], nlags)
nw_tstat1, p_values1 = fn_newey_tstat(
    coeff_reg1['lambda_CAPM'], mean_coeff_reg1['lambda_CAPM'], nlags)

# Calculate the average R-squared value
rsquared_mean_reg1 = np.nanmean(rsquared_reg1) * 100

# Prepare the res_1 for display in a DataFrame
regression_res_1 = pd.DataFrame({
    'Variable': ['Constant', 'CAPM'],
    'Coefficient (%)': [lambda_0_coef_reg1, lambda_CAPM_coef_reg1],
    't-Statistic': [nw_tstat0[0], nw_tstat1[0]],
    'p-Value': [p_values0[0], p_values1[0]],
    'R-squared (%)': [rsquared_mean_reg1, '']  # R-squared for one row
})

# Print the res_1 in tabular form
print("Regression res_1:")
print(regression_res_1.to_string(index=False))


# =============================================================================
# Reg 2   4semibetas
# =============================================================================
# Initialize lists to store the regression res_1
lambda_0_list = []
lambda_N_list = []
lambda_P_list = []
lambda_M_pos_list = []
lambda_M_neg_list = []
rsquared_list = []

# Lists for storing t-statistics and p-values
tstat_lambda_0_list = []
tstat_lambda_N_list = []
tstat_lambda_P_list = []
tstat_lambda_M_pos_list = []
tstat_lambda_M_neg_list = []

pvalue_lambda_0_list = []
pvalue_lambda_N_list = []
pvalue_lambda_P_list = []
pvalue_lambda_M_pos_list = []
pvalue_lambda_M_neg_list = []

# Loop through each share in r_monthly
for share_1 in r_monthly.columns:
    N_values = beta_N[share_1]
    P_values = beta_P[share_1]
    M_pos_values = beta_M_pos[share_1]
    M_neg_values = beta_M_neg[share_1]
    r_values = r_monthly[share_1]

    # Align the data (ensure all Win_series have the same dates and remove NaN values)
    temp_df = pd.concat([N_values, P_values, M_pos_values, M_neg_values, r_values],
                        axis=1, keys=['N', 'P', 'M_pos', 'M_neg', 'r']).dropna()

    # If there's not enough data after alignment, skip this share_1
    if len(temp_df) < 2:
        continue

    X = sm.add_constant(temp_df[['N', 'P', 'M_pos', 'M_neg']])
    y = temp_df['r']  # r_{t+1}
    model = sm.OLS(y, X).fit()

    # Store the regression res_1 (coefficients and t-statistics)
    lambda_0_list.append(model.params['const'])
    lambda_N_list.append(model.params['N'])
    lambda_P_list.append(model.params['P'])
    lambda_M_pos_list.append(model.params['M_pos'])
    lambda_M_neg_list.append(model.params['M_neg'])
    rsquared_list.append(model.rsquared)

    # Store the t-statistics
    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_N_list.append(model.tvalues['N'])
    tstat_lambda_P_list.append(model.tvalues['P'])
    tstat_lambda_M_pos_list.append(model.tvalues['M_pos'])
    tstat_lambda_M_neg_list.append(model.tvalues['M_neg'])

    # Store the p-values
    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_N_list.append(model.pvalues['N'])
    pvalue_lambda_P_list.append(model.pvalues['P'])
    pvalue_lambda_M_pos_list.append(model.pvalues['M_pos'])
    pvalue_lambda_M_neg_list.append(model.pvalues['M_neg'])

# Convert lists to DataFrames for easy analysis
res_1_df = pd.DataFrame({
    'lambda_0': lambda_0_list,
    'lambda_N': lambda_N_list,
    'lambda_P': lambda_P_list,
    'lambda_M_pos': lambda_M_pos_list,
    'lambda_M_neg': lambda_M_neg_list,
    'rsquared': rsquared_list
})

tstats_df = pd.DataFrame({
    'tstat_lambda_0': tstat_lambda_0_list,
    'tstat_lambda_N': tstat_lambda_N_list,
    'tstat_lambda_P': tstat_lambda_P_list,
    'tstat_lambda_M_pos': tstat_lambda_M_pos_list,
    'tstat_lambda_M_neg': tstat_lambda_M_neg_list
})

pvalues_df = pd.DataFrame({
    'pvalue_lambda_0': pvalue_lambda_0_list,
    'pvalue_lambda_N': pvalue_lambda_N_list,
    'pvalue_lambda_P': pvalue_lambda_P_list,
    'pvalue_lambda_M_pos': pvalue_lambda_M_pos_list,
    'pvalue_lambda_M_neg': pvalue_lambda_M_neg_list
})

# Calculate the average of the regression coefficients, t-statistics, and p-values
mean_res_1 = res_1_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

print("Average Regression res_1:")
print(mean_res_1)
print("\nAverage t-Statistics:")
print(mean_tstats)
print("\nAverage p-Values:")
print(mean_pvalues)


# =============================================================================
# Reg 4 fama factors with semibetas
# =============================================================================

# Initialize lists to store regression res_1
lambda_0_reg3 = []
lambda_N_reg3 = []
lambda_P_reg3 = []
lambda_M_pos_reg3 = []
lambda_M_neg_reg3 = []
lambda_smb_reg3 = []
lambda_hml_reg3 = []
lambda_umd_reg3 = []
rsquared_reg3 = []

# Track the number of regressions run
regression_counts_reg3 = {}

# Loop through each share_1 (PERMNO) in r_monthly
for share_1 in r_monthly.columns:
    N_values = beta_N[share_1]
    P_values = beta_P[share_1]
    M_pos_values = beta_M_pos[share_1]
    M_neg_values = beta_M_neg[share_1]
    r_values = r_monthly[share_1]

    temp_df = pd.concat([N_values, P_values, M_pos_values, M_neg_values, r_values,
                         ff_factors_monthly['smb'], ff_factors_monthly['hml'], ff_factors_monthly['umd']],
                        axis=1, keys=['N', 'P', 'M_pos', 'M_neg', 'r', 'smb', 'hml', 'umd']).dropna()

    if temp_df.empty or len(temp_df) < 2:
        print(f"Skipping share_1 {share_1} due to insufficient data.")
        continue

    scaler = StandardScaler()
    independent_vars = temp_df[[
        'N', 'P', 'M_pos', 'M_neg', 'smb', 'hml', 'umd']]
    independent_vars_scaled = pd.DataFrame(scaler.fit_transform(independent_vars),
                                           columns=independent_vars.columns, index=independent_vars.index)

    temp_df['r'] = scaler.fit_transform(temp_df[['r']])

    x = sm.add_constant(independent_vars_scaled)
    y = temp_df['r']

    model_reg3 = sm.OLS(y, x).fit()

    lambda_0_reg3.append(model_reg3.params.get('const', np.nan))
    lambda_N_reg3.append(model_reg3.params.get('N', np.nan))
    lambda_P_reg3.append(model_reg3.params.get('P', np.nan))
    lambda_M_pos_reg3.append(model_reg3.params.get('M_pos', np.nan))
    lambda_M_neg_reg3.append(model_reg3.params.get('M_neg', np.nan))
    lambda_smb_reg3.append(model_reg3.params.get('smb', np.nan))
    lambda_hml_reg3.append(model_reg3.params.get('hml', np.nan))
    lambda_umd_reg3.append(model_reg3.params.get('umd', np.nan))
    rsquared_reg3.append(model_reg3.rsquared)

    regression_counts_reg3[share_1] = len(temp_df)

coeff_reg3 = pd.DataFrame({
    'lambda_0': lambda_0_reg3,
    'lambda_N': lambda_N_reg3,
    'lambda_P': lambda_P_reg3,
    'lambda_M_pos': lambda_M_pos_reg3,
    'lambda_M_neg': lambda_M_neg_reg3,
    'lambda_smb': lambda_smb_reg3,
    'lambda_hml': lambda_hml_reg3,
    'lambda_umd': lambda_umd_reg3
})

mean_coeff_reg3 = coeff_reg3.mean()

lambda_0_coef_reg3 = mean_coeff_reg3['lambda_0'] * 100
lambda_N_coef_reg3 = mean_coeff_reg3['lambda_N'] * 100
lambda_P_coef_reg3 = mean_coeff_reg3['lambda_P'] * 100
lambda_M_pos_coef_reg3 = mean_coeff_reg3['lambda_M_pos'] * 100
lambda_M_neg_coef_reg3 = mean_coeff_reg3['lambda_M_neg'] * 100
lambda_smb_coef_reg3 = mean_coeff_reg3['lambda_smb'] * 100
lambda_hml_coef_reg3 = mean_coeff_reg3['lambda_hml'] * 100
lambda_umd_coef_reg3 = mean_coeff_reg3['lambda_umd'] * 100

nlags = round(0.75 * len(r_monthly) ** (1/3))
nw_tstat0_reg3, p_values0_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_0']/10, mean_coeff_reg3['lambda_0'], nlags)
nw_tstat_N_reg3, p_values_N_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_N'], mean_coeff_reg3['lambda_N'], nlags)
nw_tstat_P_reg3, p_values_P_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_P'], mean_coeff_reg3['lambda_P'], nlags)
nw_tstat_M_pos_reg3, p_values_M_pos_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_M_pos'], mean_coeff_reg3['lambda_M_pos'], nlags)
nw_tstat_M_neg_reg3, p_values_M_neg_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_M_neg'], mean_coeff_reg3['lambda_M_neg'], nlags)
nw_tstat_smb_reg3, p_values_smb_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_smb'], mean_coeff_reg3['lambda_smb'], nlags)
nw_tstat_hml_reg3, p_values_hml_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_hml'], mean_coeff_reg3['lambda_hml'], nlags)
nw_tstat_umd_reg3, p_values_umd_reg3 = fn_newey_tstat(
    coeff_reg3['lambda_umd'], mean_coeff_reg3['lambda_umd'], nlags)


rsquared_mean_reg3 = np.mean(rsquared_reg3) * 100

regression_res_1_reg3 = pd.DataFrame({
    'Variable': ['Constant', 'N', 'P', 'M_pos', 'M_neg', 'smb', 'hml', 'umd'],
    'Coefficient (%)': [lambda_0_coef_reg3, lambda_N_coef_reg3, lambda_P_coef_reg3,
                        lambda_M_pos_coef_reg3, lambda_M_neg_coef_reg3,
                        lambda_smb_coef_reg3, lambda_hml_coef_reg3, lambda_umd_coef_reg3],
    't-Statistic': [nw_tstat0_reg3[0], nw_tstat_N_reg3[0], nw_tstat_P_reg3[0],
                    nw_tstat_M_pos_reg3[0], nw_tstat_M_neg_reg3[0],
                    nw_tstat_smb_reg3[0], nw_tstat_hml_reg3[0], nw_tstat_umd_reg3[0]],
    'p-Value': [p_values0_reg3[0], p_values_N_reg3[0], p_values_P_reg3[0],
                p_values_M_pos_reg3[0], p_values_M_neg_reg3[0],
                p_values_smb_reg3[0], p_values_hml_reg3[0], p_values_umd_reg3[0]],
    # R-squared only in the first row
    'R-squared (%)': [rsquared_mean_reg3, '', '', '', '', '', '', '']
})

print("Regression res_1 with Standardized Fama-French and Semibetas:")
print(regression_res_1_reg3.to_string(index=False))


# =============================================================================
# reg 4 beta up and down
# =============================================================================

lambda_0_list = []
lambda_up_list = []
lambda_down_list = []
rsquared_list = []

tstat_lambda_0_list = []
tstat_lambda_up_list = []
tstat_lambda_down_list = []

pvalue_lambda_0_list = []
pvalue_lambda_up_list = []
pvalue_lambda_down_list = []

for share_1 in r_monthly.columns:
    up_values = beta_up[share_1]
    down_values = beta_down[share_1]
    r_values = r_monthly[share_1]

    temp_df = pd.concat([up_values, down_values, r_values],
                        axis=1, keys=['up', 'down', 'r']).dropna()

    if len(temp_df) < 2:
        continue

    X = sm.add_constant(temp_df[['up', 'down']])
    y = temp_df['r']  # r_{t+1}

    model = sm.OLS(y, X).fit()

    lambda_0_list.append(model.params['const'])
    lambda_up_list.append(model.params['up'])
    lambda_down_list.append(model.params['down'])
    rsquared_list.append(model.rsquared)

    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_up_list.append(model.tvalues['up'])
    tstat_lambda_down_list.append(model.tvalues['down'])

    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_up_list.append(model.pvalues['up'])
    pvalue_lambda_down_list.append(model.pvalues['down'])

res_1_df = pd.DataFrame({
    'lambda_0': lambda_0_list,
    'lambda_up': lambda_up_list,
    'lambda_down': lambda_down_list,
    'rsquared': rsquared_list
})

tstats_df = pd.DataFrame({
    'tstat_lambda_0': tstat_lambda_0_list,
    'tstat_lambda_up': tstat_lambda_up_list,
    'tstat_lambda_down': tstat_lambda_down_list
})

pvalues_df = pd.DataFrame({
    'pvalue_lambda_0': pvalue_lambda_0_list,
    'pvalue_lambda_up': pvalue_lambda_up_list,
    'pvalue_lambda_down': pvalue_lambda_down_list
})

mean_res_1 = res_1_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

print("Average Regression res_1:")
print(mean_res_1)
print("\nAverage t-Statistics:")
print(mean_tstats)
print("\nAverage p-Values:")
print(mean_pvalues)


# =============================================================================
# reg 5 beta up down with semibeta
# =============================================================================
lambda_0_list = []
lambda_up_list = []
lambda_down_list = []
lambda_N_list = []
lambda_P_list = []
lambda_M_pos_list = []
lambda_M_neg_list = []
rsquared_list = []

tstat_lambda_0_list = []
tstat_lambda_up_list = []
tstat_lambda_down_list = []
tstat_lambda_N_list = []
tstat_lambda_P_list = []
tstat_lambda_M_pos_list = []
tstat_lambda_M_neg_list = []

pvalue_lambda_0_list = []
pvalue_lambda_up_list = []
pvalue_lambda_down_list = []
pvalue_lambda_N_list = []
pvalue_lambda_P_list = []
pvalue_lambda_M_pos_list = []
pvalue_lambda_M_neg_list = []

# Loop through each share_1 (PERMNO) in r_monthly
for share_1 in r_monthly.columns:
    # Extract the data for the current share_1 from all DataFrames
    up_values = beta_up[share_1]
    down_values = beta_down[share_1]
    N_values = beta_N[share_1]
    P_values = beta_P[share_1]
    M_pos_values = beta_M_pos[share_1]
    M_neg_values = beta_M_neg[share_1]
    r_values = r_monthly[share_1]

    # Align the data (ensure all Win_series have the same dates and remove NaN values)
    temp_df = pd.concat([up_values, down_values, N_values, P_values, M_pos_values, M_neg_values, r_values],
                        axis=1, keys=['up', 'down', 'N', 'P', 'M_pos', 'M_neg', 'r']).dropna()

    # If there's not enough data after alignment, skip this share
    if len(temp_df) < 2:
        continue

    X = sm.add_constant(temp_df[['up', 'down', 'N', 'P', 'M_pos', 'M_neg']])
    y = temp_df['r']
    model = sm.OLS(y, X).fit()

    lambda_0_list.append(model.params['const'])
    lambda_up_list.append(model.params['up'])
    lambda_down_list.append(model.params['down'])
    lambda_N_list.append(model.params['N'])
    lambda_P_list.append(model.params['P'])
    lambda_M_pos_list.append(model.params['M_pos'])
    lambda_M_neg_list.append(model.params['M_neg'])
    rsquared_list.append(model.rsquared)

    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_up_list.append(model.tvalues['up'])
    tstat_lambda_down_list.append(model.tvalues['down'])
    tstat_lambda_N_list.append(model.tvalues['N'])
    tstat_lambda_P_list.append(model.tvalues['P'])
    tstat_lambda_M_pos_list.append(model.tvalues['M_pos'])
    tstat_lambda_M_neg_list.append(model.tvalues['M_neg'])

    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_up_list.append(model.pvalues['up'])
    pvalue_lambda_down_list.append(model.pvalues['down'])
    pvalue_lambda_N_list.append(model.pvalues['N'])
    pvalue_lambda_P_list.append(model.pvalues['P'])
    pvalue_lambda_M_pos_list.append(model.pvalues['M_pos'])
    pvalue_lambda_M_neg_list.append(model.pvalues['M_neg'])

res_1_df = pd.DataFrame({
    'lambda_0': lambda_0_list,
    'lambda_up': lambda_up_list,
    'lambda_down': lambda_down_list,
    'lambda_N': lambda_N_list,
    'lambda_P': lambda_P_list,
    'lambda_M_pos': lambda_M_pos_list,
    'lambda_M_neg': lambda_M_neg_list,
    'rsquared': rsquared_list
})

tstats_df = pd.DataFrame({
    'tstat_lambda_0': tstat_lambda_0_list,
    'tstat_lambda_up': tstat_lambda_up_list,
    'tstat_lambda_down': tstat_lambda_down_list,
    'tstat_lambda_N': tstat_lambda_N_list,
    'tstat_lambda_P': tstat_lambda_P_list,
    'tstat_lambda_M_pos': tstat_lambda_M_pos_list,
    'tstat_lambda_M_neg': tstat_lambda_M_neg_list
})

pvalues_df = pd.DataFrame({
    'pvalue_lambda_0': pvalue_lambda_0_list,
    'pvalue_lambda_up': pvalue_lambda_up_list,
    'pvalue_lambda_down': pvalue_lambda_down_list,
    'pvalue_lambda_N': pvalue_lambda_N_list,
    'pvalue_lambda_P': pvalue_lambda_P_list,
    'pvalue_lambda_M_pos': pvalue_lambda_M_pos_list,
    'pvalue_lambda_M_neg': pvalue_lambda_M_neg_list
})

mean_res_1 = res_1_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

# Print the average regression res_1, t-statistics, and p-values
print("Average Regression res_1 for Regression 5:")
print(mean_res_1)
print("\nAverage t-Statistics for Regression 5:")
print(mean_tstats)
print("\nAverage p-Values for Regression 5:")
print(mean_pvalues)


# =============================================================================
# Reg6 coskew cokurt
# =============================================================================
lambda_0_list = []
lambda_coskewness_list = []
lambda_cokurtosis_list = []
rsquared_list = []

tstat_lambda_0_list = []
tstat_lambda_coskewness_list = []
tstat_lambda_cokurtosis_list = []

pvalue_lambda_0_list = []
pvalue_lambda_coskewness_list = []
pvalue_lambda_cokurtosis_list = []

for share_1 in r_monthly.columns:
    # Extract the data for the current share_1 from all DataFrames
    if share_1 not in coskewness.columns or share_1 not in cokurtosis.columns:
        continue

    coskewness_values = coskewness[share_1]
    cokurtosis_values = cokurtosis[share_1]
    r_values = r_monthly[share_1]

    temp_df = pd.concat([coskewness_values, cokurtosis_values, r_values],
                        axis=1, keys=['coskewness', 'cokurtosis', 'r']).dropna()

    if len(temp_df) < 2:
        continue

    X = sm.add_constant(temp_df[['coskewness', 'cokurtosis']])
    y = temp_df['r']  # r_{t+1}

    model = sm.OLS(y, X).fit()
    lambda_0_list.append(model.params['const'])
    lambda_coskewness_list.append(model.params['coskewness'])
    lambda_cokurtosis_list.append(model.params['cokurtosis'])
    rsquared_list.append(model.rsquared)
    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_coskewness_list.append(model.tvalues['coskewness'])
    tstat_lambda_cokurtosis_list.append(model.tvalues['cokurtosis'])
    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_coskewness_list.append(model.pvalues['coskewness'])
    pvalue_lambda_cokurtosis_list.append(model.pvalues['cokurtosis'])

res_1_df = pd.DataFrame({
    'lambda_0': lambda_0_list,
    'lambda_coskewness': lambda_coskewness_list,
    'lambda_cokurtosis': lambda_cokurtosis_list,
    'rsquared': rsquared_list
})

tstats_df = pd.DataFrame({
    'tstat_lambda_0': tstat_lambda_0_list,
    'tstat_lambda_coskewness': tstat_lambda_coskewness_list,
    'tstat_lambda_cokurtosis': tstat_lambda_cokurtosis_list
})

pvalues_df = pd.DataFrame({
    'pvalue_lambda_0': pvalue_lambda_0_list,
    'pvalue_lambda_coskewness': pvalue_lambda_coskewness_list,
    'pvalue_lambda_cokurtosis': pvalue_lambda_cokurtosis_list
})

mean_res_1 = res_1_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

print("Average Regression res_1 for Regression 6:")
print(mean_res_1)
print("\nAverage t-Statistics for Regression 6:")
print(mean_tstats)
print("\nAverage p-Values for Regression 6:")
print(mean_pvalues)


# =============================================================================
# reg7 coskew cokurt with semibeta
# =============================================================================

lambda_0_list = []
lambda_coskewness_list = []
lambda_cokurtosis_list = []
lambda_N_list = []
lambda_P_list = []
lambda_M_pos_list = []
lambda_M_neg_list = []
rsquared_list = []

tstat_lambda_0_list = []
tstat_lambda_coskewness_list = []
tstat_lambda_cokurtosis_list = []
tstat_lambda_N_list = []
tstat_lambda_P_list = []
tstat_lambda_M_pos_list = []
tstat_lambda_M_neg_list = []

pvalue_lambda_0_list = []
pvalue_lambda_coskewness_list = []
pvalue_lambda_cokurtosis_list = []
pvalue_lambda_N_list = []
pvalue_lambda_P_list = []
pvalue_lambda_M_pos_list = []
pvalue_lambda_M_neg_list = []

# Loop through each share_1 (PERMNO) in r_monthly
for share_1 in r_monthly.columns:
    # Extract the data for the current share_1 from all DataFrames
    if share_1 not in coskewness.columns or share_1 not in cokurtosis.columns or \
       share_1 not in beta_N.columns or share_1 not in beta_P.columns or \
       share_1 not in beta_M_pos.columns or share_1 not in beta_M_neg.columns:
        continue

    coskewness_values = coskewness[share_1]
    cokurtosis_values = cokurtosis[share_1]
    N_values = beta_N[share_1]
    P_values = beta_P[share_1]
    M_pos_values = beta_M_pos[share_1]
    M_neg_values = beta_M_neg[share_1]
    r_values = r_monthly[share_1]

    # Align the data (ensure all Win_series have the same dates and remove NaN values)
    temp_df = pd.concat([coskewness_values, cokurtosis_values, N_values, P_values, M_pos_values, M_neg_values, r_values],
                        axis=1, keys=['coskewness', 'cokurtosis', 'N', 'P', 'M_pos', 'M_neg', 'r']).dropna()

    # If there's not enough data after alignment, skip this share_1
    if len(temp_df) < 2:
        continue

    X = sm.add_constant(
        temp_df[['coskewness', 'cokurtosis', 'N', 'P', 'M_pos', 'M_neg']])
    y = temp_df['r']  # r_{t+1}

    model = sm.OLS(y, X).fit()
    lambda_0_list.append(model.params['const'])
    lambda_coskewness_list.append(model.params['coskewness'])
    lambda_cokurtosis_list.append(model.params['cokurtosis'])
    lambda_N_list.append(model.params['N'])
    lambda_P_list.append(model.params['P'])
    lambda_M_pos_list.append(model.params['M_pos'])
    lambda_M_neg_list.append(model.params['M_neg'])
    rsquared_list.append(model.rsquared)
    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_coskewness_list.append(model.tvalues['coskewness'])
    tstat_lambda_cokurtosis_list.append(model.tvalues['cokurtosis'])
    tstat_lambda_N_list.append(model.tvalues['N'])
    tstat_lambda_P_list.append(model.tvalues['P'])
    tstat_lambda_M_pos_list.append(model.tvalues['M_pos'])
    tstat_lambda_M_neg_list.append(model.tvalues['M_neg'])
    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_coskewness_list.append(model.pvalues['coskewness'])
    pvalue_lambda_cokurtosis_list.append(model.pvalues['cokurtosis'])
    pvalue_lambda_N_list.append(model.pvalues['N'])
    pvalue_lambda_P_list.append(model.pvalues['P'])
    pvalue_lambda_M_pos_list.append(model.pvalues['M_pos'])
    pvalue_lambda_M_neg_list.append(model.pvalues['M_neg'])

# Convert lists to DataFrames for easy analysis
res_1_df = pd.DataFrame({
    'lambda_0': lambda_0_list,
    'lambda_coskewness': lambda_coskewness_list,
    'lambda_cokurtosis': lambda_cokurtosis_list,
    'lambda_N': lambda_N_list,
    'lambda_P': lambda_P_list,
    'lambda_M_pos': lambda_M_pos_list,
    'lambda_M_neg': lambda_M_neg_list,
    'rsquared': rsquared_list
})

tstats_df = pd.DataFrame({
    'tstat_lambda_0': tstat_lambda_0_list,
    'tstat_lambda_coskewness': tstat_lambda_coskewness_list,
    'tstat_lambda_cokurtosis': tstat_lambda_cokurtosis_list,
    'tstat_lambda_N': tstat_lambda_N_list,
    'tstat_lambda_P': tstat_lambda_P_list,
    'tstat_lambda_M_pos': tstat_lambda_M_pos_list,
    'tstat_lambda_M_neg': tstat_lambda_M_neg_list
})

pvalues_df = pd.DataFrame({
    'pvalue_lambda_0': pvalue_lambda_0_list,
    'pvalue_lambda_coskewness': pvalue_lambda_coskewness_list,
    'pvalue_lambda_cokurtosis': pvalue_lambda_cokurtosis_list,
    'pvalue_lambda_N': pvalue_lambda_N_list,
    'pvalue_lambda_P': pvalue_lambda_P_list,
    'pvalue_lambda_M_pos': pvalue_lambda_M_pos_list,
    'pvalue_lambda_M_neg': pvalue_lambda_M_neg_list
})

mean_res_1 = res_1_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

print("Average Regression res_1 for Regression with Coskewness, Cokurtosis, and Semibetas:")
print(mean_res_1)
print("\nAverage t-Statistics for Regression:")
print(mean_tstats)
print("\nAverage p-Values for Regression:")
print(mean_pvalues)


# =============================================================================
# Figures- density of semibetas and correlation heatmap
# =============================================================================

# density plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.kdeplot(beta_CAPM.values.flatten(),
            label=r'$\beta$', color='#FF6347', lw=2)
sns.kdeplot(beta_N.values.flatten(), label=r'$\beta^N$',
            linestyle='--', lw=2, color='#1E90FF')
sns.kdeplot(beta_P.values.flatten(), label=r'$\beta^P$',
            linestyle='-.', lw=2, color='#FFA500')
sns.kdeplot(beta_M_pos.values.flatten(),
            label=r'$\beta^{M+}$', linestyle=':', lw=2, color='#32CD32')
sns.kdeplot(beta_M_neg.values.flatten(),
            label=r'$\beta^{M-}$', linestyle='-', lw=2, color='#00CED1')

plt.xlim(-1, 3)  # Adjust this based on your data distribution

plt.title("Panel A: Distribution of Semibetas", fontsize=16, fontweight='bold')
plt.xlabel(r'$\beta$', fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine()

plt.tight_layout()
plt.show()


# corrlation heatmap
for i in range(len(betas)):
    for j in range(i + 1, len(betas)):
        # Copy upper triangle to lower triangle
        corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i]

plt.figure(figsize=(10, 6))  # Set the size of the heatmap
sns.heatmap(corr_matrix.astype(float), annot=True,
            cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)

# Adding title and labels
plt.title("Correlation Heatmap of Semibetas")
plt.show()


# wald test

# Defining the null hypothesis constraints matrix
# The constraints matrix should be constructed based on the hypothesis H0: lambda_N = lambda_P = -lambda_M_pos = -lambda_M_neg
R = np.array([
    [1, -1, 0, 0],    # Tests lambda_N = lambda_P
    [1, 0, 1, 0],    # Tests lambda_N = -lambda_M_pos
    [1, 0, 0, 1]     # Tests lambda_N = -lambda_M_neg
])

# Mean values of the coefficients from the res_1
mean_coefficients = np.array([
    mean_res_1['lambda_N'],
    mean_res_1['lambda_P'],
    mean_res_1['lambda_M_pos'],
    mean_res_1['lambda_M_neg']
])

cov_matrix = np.cov([lambda_N_list, lambda_P_list,
                    lambda_M_pos_list, lambda_M_neg_list])

# Calculate the Wald statistic
diff = R @ mean_coefficients  # (R*Beta)
# (R*Beta)' (R*Cov*R')^-1 (R*Beta)
wald_stat = diff.T @ np.linalg.inv(R @ cov_matrix @ R.T) @ diff

# Degrees of freedom is the number of constraints (rows in matrix R)
df = R.shape[0]

# Calculate the p-value for the Wald test statistic
p_value = chi2.sf(wald_stat, df)

print("Wald Test Statistic:", wald_stat)
print("Degrees of Freedom:", df)
print("p-Value:", p_value)
