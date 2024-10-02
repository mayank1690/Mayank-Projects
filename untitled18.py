# final monthly on labcleaned


# --final monthly

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


def handle_outlier_months(daily_rtn_df):
    for stock in daily_rtn_df.columns:
        monthly_count = daily_rtn_df[stock].resample('ME').count()
        months_to_nan = monthly_count[monthly_count < 10].dropna().index
        daily_rtn_df.loc[daily_rtn_df.index.to_period('M').isin(
            months_to_nan.to_period('M')), stock] = np.nan
    return daily_rtn_df


# Semibeta calculation- calculates monthly semibeta
def semibeta(r_signed, m_signed, m_total):
    num_df = r_signed * m_signed.values
    denom_df = m_total ** 2
    sum_num_df = num_df.resample('ME').sum(min_count=1)
    sum_denom_df = denom_df.resample('ME').sum(min_count=1)
    semibeta_df = sum_num_df / sum_denom_df.values
    return semibeta_df


def semibeta_mix(r_signed, m_signed, m_total):
    num_df = r_signed * m_signed.values
    denom_df = m_total ** 2
    sum_num_df = num_df.resample('ME').sum(min_count=1)
    sum_denom_df = denom_df.resample('ME').sum(min_count=1)
    semibeta_df = -sum_num_df / sum_denom_df.values  # sign change
    return semibeta_df

# Function to winsorize a single column


def winsorize_col(series, lower_percentile, upper_percentile):
    return pd.Series(winsorize(series, limits=(lower_percentile, upper_percentile)), index=series.index)

# Semibeta summary statistics


def semibeta_sumstat(semibeta_df):
    cross_mean = semibeta_df.mean(axis=1)
    cross_std = semibeta_df.std(axis=1, ddof=1)
    cross_median = semibeta_df.median(axis=1)

    ts_mean = cross_mean.mean()
    ts_std = cross_std.mean()
    ts_median = cross_median.mean()

    return ts_mean, ts_median, ts_std

# Time-series mean of cross-sectional correlation


def average_cross_sec_corr(matrix1, matrix2):
    correlations = []
    for i in range(len(matrix1)):
        corr = matrix1.iloc[i].corr(matrix2.iloc[i], 'pearson')
        correlations.append(corr)
    return np.mean(correlations)


def newey_west_se(lambdas, nlags):
    results = sm.OLS(lambdas, np.ones(len(lambdas))).fit()
    nw_cov = cov_hac(results, nlags=nlags)
    nw_se = np.sqrt(np.diag(nw_cov))
    return nw_se


def newey_west_tstat(lambdas, mean_lambdas, nlags):
    nw_se = newey_west_se(lambdas, nlags)
    nw_tstat = mean_lambdas / nw_se
    p_values = 2 * (1 - norm.cdf(np.abs(nw_tstat)))
    return nw_tstat, p_values


def semibeta_up(r_all, m_signed):
    num_df = r_all * m_signed.values
    denom_df = m_signed ** 2
    sum_num_df = num_df.resample('ME').sum(min_count=1)
    sum_denom_df = denom_df.resample('ME').sum(min_count=1)
    beta_up = sum_num_df / sum_denom_df.values
    return beta_up


def semibeta_down(r_all, m_signed):
    num_df = r_all * m_signed.values
    denom_df = m_signed ** 2
    sum_num_df = num_df.resample('ME').sum(min_count=1)
    sum_denom_df = denom_df.resample('ME').sum(min_count=1)
    beta_up = sum_num_df / sum_denom_df.values
    return beta_up


def calculate_coskewness_cokurtosis(r, m):
    # Step 1: Resample the data by month and calculate the mean for each stock and the market
    r_monthly_mean = r.resample('M').mean()
    m_monthly_mean = m.resample('M').mean()

    # Step 2: Calculate the daily deviations from the monthly mean
    r_diff = r.groupby(r.index.to_period(
        'M')).transform(lambda x: x - x.mean())
    m_diff = m.groupby(m.index.to_period(
        'M')).transform(lambda x: x - x.mean())

    # Step 3: Calculate the components for coskewness and cokurtosis
    # Coskewness numerator: (r_diff * (m_diff^2)).sum() for each month
    coskewness_num = r_diff.mul(m_diff.values**2, axis=0).resample('M').sum()

    # Coskewness denominator: (r_diff^2).sum() * (m_diff^2).sum() for each month
    coskewness_denom_r = r_diff.pow(2).resample(
        'M').sum()  # r_diff^2 sum per month
    coskewness_denom_m = m_diff.pow(2).resample(
        'M').sum()  # m_diff^2 sum per month

    coskewness = coskewness_num / \
        (coskewness_denom_r * coskewness_denom_m.values +
         1e-8)  # Avoid division by zero

    # Cokurtosis numerator: (r_diff * (m_diff^3)).sum() for each month
    cokurtosis_num = r_diff.mul(m_diff.values**3, axis=0).resample('M').sum()

    # Cokurtosis denominator: ((r_diff^2).sum() * (m_diff^2).sum())^(3/2) for each month
    cokurtosis_denom_r = r_diff.pow(2).resample('M').sum()
    cokurtosis_denom_m = m_diff.pow(2).resample('M').sum()
    cokurtosis_denom = (cokurtosis_denom_r *
                        cokurtosis_denom_m.values).pow(3/2)

    cokurtosis = cokurtosis_num / \
        (cokurtosis_denom + 1e-8)  # Avoid division by zero

    return coskewness, cokurtosis


# =============================================================================
# # Define the file path and chunk size
# file_path = r"C:\\Users\\mayan\\OneDrive - University of Birmingham\\Desktop\\dessertation\\data\\CRPSsmall.csv"
# thedata = pd.read_csv(file_path)
#
# thedata['date'] = pd.to_datetime(thedata['date'])
#
# # Set 'date' as the index without removing the column
# thedata.set_index('date', drop=False, inplace=True)
#
#
# # Extract relevant columns
# relevant_columns = ['PRC', 'VOL', 'SHROUT', 'DLSTCD']
#
# # Convert the columns to numeric types without coercing errors and take absolute values of 'PRC'
# thedata['PRC'] = thedata['PRC'].abs()  # Take absolute values of PRC
# thedata[relevant_columns] = thedata[relevant_columns].apply(pd.to_numeric)
#
# # Identify PERMNOs where PRC is less than 5
# permno_to_drop = thedata[thedata['PRC'] < 5]['PERMNO'].unique()
#
# # Drop rows where PERMNO is in the identified list
# thedata = thedata[~thedata['PERMNO'].isin(permno_to_drop)]
#
# # Ensure DLSTCD is numeric and relevant columns are prepared
# thedata['DLSTCD'] = pd.to_numeric(thedata['DLSTCD'])
#
# # Drop rows where DLSTCD has a value (i.e., not null)
# thedata = thedata[thedata['DLSTCD'].isna()]
#
# # Group by 'PERMNO' and calculate the average trading volume ('VOL')
# average_volume = thedata.groupby('PERMNO')['VOL'].mean()
#
# # Calculate the 15th percentile of the average trading volume
# lower_15_percentile = average_volume.quantile(0.15)
#
# # Identify stocks whose average trading volume is below the 15th percentile
# low_volume_stocks = average_volume[average_volume < lower_15_percentile].index
#
# # Drop rows corresponding to these low-volume stocks
# thedata = thedata[~thedata['PERMNO'].isin(low_volume_stocks)]
#
# # Report the number of such stocks dropped
# number_of_low_volume_stocks = len(low_volume_stocks)
#
# # Output the number of dropped stocks
# print(f"Number of stocks dropped with average trading volume below the 15th percentile (dropped): {
#       number_of_low_volume_stocks}")
#
# # Get the total number of unique dates in the dataset
# unique_dates = thedata['date'].unique()
#
# # Identify PERMNOs that either:
# # 1. Don't have entries for all unique dates
# # 2. Are missing PRC values for any date
# permno_to_drop = thedata.groupby('PERMNO').filter(
#     lambda x: len(x['date'].unique()) < len(
#         unique_dates) or x['PRC'].isna().any()
# )['PERMNO'].unique()
#
# # Report the number of PERMNOs to be dropped
# number_of_permno_dropped = len(permno_to_drop)
# print(f"Number of PERMNOs to be dropped: {number_of_permno_dropped}")
#
# # Drop rows corresponding to these PERMNOs
# thedata = thedata[~thedata['PERMNO'].isin(permno_to_drop)]
#
# # Count the remaining unique PERMNOs in the data
# remaining_permno_count = thedata['PERMNO'].nunique()
#
# # Report the remaining unique PERMNO count
# print(f"Number of unique PERMNOs remaining in the dataset: {
#       remaining_permno_count}")
#
# thedata.drop(columns=['date'], inplace=True)
#
# # Now reset the index to move 'date' from the index back into a column
# thedata.reset_index(inplace=True)
#
# # Step 1: Sort the data by PERMNO and date to ensure chronological order
# thedata = thedata.sort_values(by=['PERMNO', 'date'])
#
# # Step 2: Calculate daily log returns for each stock (PERMNO) based on PRC
# thedata['RET'] = thedata.groupby('PERMNO')['PRC'].apply(
#     lambda x: np.log(x / x.shift(1))
# ).reset_index(level=0, drop=True)
#
# # Step 3: Remove rows where the return ('RET') is NaN (the first date for each stock)
# thedata = thedata[~thedata['RET'].isna()]
#
#
# # Backfill 0 returns in the RET column
# thedata['RET'] = thedata['RET'].replace(0, np.nan).bfill()
#
# # Remove specified columns
# columns_to_remove = ['DLSTCD', 'VOL']
# thedata = thedata.drop(columns=columns_to_remove)
#
#
# famafrenchdata = pd.read_csv(
#     r"C:\\Users\\mayan\\OneDrive - University of Birmingham\\Desktop\\dessertation\\data\\fama fench data 1.csv")
#
# # Convert 'date' column to datetime format
# famafrenchdata['date'] = pd.to_datetime(famafrenchdata['date'])
#
# # Extract only the relevant columns ('smb', 'hml', 'umd', 'date')
# ff_factors = famafrenchdata[['date', 'smb', 'hml', 'umd']]
#
# # Ensure the 'date' column in 'thedata' is also in datetime format (if it isn't already)
# thedata['date'] = pd.to_datetime(thedata['date'])
#
# # Merge Fama-French factors into 'thedata' on 'date'
# # This will replicate the factors across all stocks for the same date
# thedata = pd.merge(thedata, ff_factors, on='date', how='left')
#
#
# # Drop the specified columns
# columns_to_drop = ['PERMCO', 'CUSIP', 'RCRDDT', 'SHRCD', 'vwretd', 'ewretd']
# thedata = thedata.drop(columns=columns_to_drop)
#
# # Calculate market capitalization for each stock
# thedata['market_cap'] = thedata['PRC'] * thedata['SHROUT']
#
# # Calculate daily market return by weighting each stock's return by its market capitalization
# daily_market_return = thedata.groupby('date').apply(
#     lambda x: np.sum(x['RET'] * x['market_cap']) / np.sum(x['market_cap']))
#
# # Add the daily market return to the dataset
# thedata['market_return'] = thedata['date'].map(daily_market_return)
#
# print(f"Remaining number of unique stocks: {
#       thedata['PERMNO'].nunique()}")
# =============================================================================


# Define the file path and chunk size
file_path = r"C:\Users\mayan\Downloads\lab_cleanedthedata.csv"
thedata = pd.read_csv(file_path)

thedata['date'] = pd.to_datetime(thedata['date'])

# Set 'date' as the index without removing the column
thedata.set_index('date', drop=False, inplace=True)

thedata.drop(columns=['date'], inplace=True)


# -----construct r and m


# Ensure 'date' is in datetime format if it's not already
thedata['date'] = pd.to_datetime(thedata['date'])

# Filter relevant columns
temp_df = thedata[['PERMNO', 'date', 'RET']]
# Pivot the data to create a DataFrame where each PERMNO is a column
# and the index is the date
r = temp_df.pivot_table(index='date', columns='PERMNO', values='RET')


# Step 2: Filter relevant columns: 'Date' and 'market_return'
m = thedata[['date', 'market_return']]

# Step 3: Remove any duplicates (if necessary) and keep only one entry per date
m = m.drop_duplicates(subset=['date'])

# Step 4: Set 'Date' as index
m.set_index('date', inplace=True)


# Rename the 'market_return' column to something more meaningful, e.g., 'I.201'
m.columns = ['daily market ret']


r_monthly = r.resample('M').sum()


# --------------calculating factors

r_pos = r.mask(r < 0, 0)
r_neg = r.mask(r > 0, 0)
m_pos = m.mask(m < 0, 0)
m_neg = m.mask(m > 0, 0)

# Handle outlier months
r = handle_outlier_months(r)
m = handle_outlier_months(m)
r_pos = handle_outlier_months(r_pos)
r_neg = handle_outlier_months(r_neg)
m_pos = handle_outlier_months(m_pos)
m_neg = handle_outlier_months(m_neg)


# semibeta calculation
# Calculate monthly semibetas
beta_CAPM = semibeta(r, m, m)
beta_N = semibeta(r_neg, m_neg, m)
beta_P = semibeta(r_pos, m_pos, m)
beta_M_pos = semibeta_mix(r_neg, m_pos, m)
beta_M_neg = semibeta_mix(r_pos, m_neg, m)

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
    element.apply(winsorize_col, lower_percentile=0.01,
                  upper_percentile=0.01, axis=0)

# Convert all column names to strings and indices to PeriodIndex
dataframes_to_convert = [beta_up, beta_down, r_monthly, beta_CAPM, beta_N,
                         beta_P, beta_M_pos, beta_M_neg, coskewness, cokurtosis, ff_factors_monthly]

for df in dataframes_to_convert:
    df.columns = df.columns.astype(str)
    df.index = pd.PeriodIndex(df.index, freq='M')


# =============================================================================
# # Ensuring all DataFrames have the same columns (PERMNOs) and indices (dates)
# def align_dataframes(base_df, *dfs):
#     common_columns = base_df.columns
#     common_indices = base_df.index
#     return [df[common_columns].loc[common_indices] for df in dfs]
#
# r_monthly, beta_up, beta_down = align_dataframes(r_monthly, beta_up, beta_down)
# beta_N, beta_P, beta_M_pos, beta_M_neg = align_dataframes(r_monthly, beta_N, beta_P, beta_M_pos, beta_M_neg)
# coskewness, cokurtosis = align_dataframes(r_monthly, coskewness, cokurtosis)
#
#
# =============================================================================

# --------------semibeta summary stats
# Semibeta summary statistics #table 1
sumstat_index = ['Mean', 'Med', 'Std']
sumstat_col = ['B', 'B_N', 'B_P', 'B_M+', 'B_M-']
sumstat = pd.DataFrame(index=sumstat_index, columns=sumstat_col)
sumstat['B'] = semibeta_sumstat(beta_CAPM)
sumstat['B_N'] = semibeta_sumstat(beta_N)
sumstat['B_P'] = semibeta_sumstat(beta_P)
sumstat['B_M+'] = semibeta_sumstat(beta_M_pos)
sumstat['B_M-'] = semibeta_sumstat(beta_M_neg)

# Correlation between semibetas #table 1
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


# print table 1
# Printing the Semibeta Summary Statistics
print("Semibeta Summary Statistics")
print(sumstat.to_string())
# Printing the Correlation Matrix
print("\nSemibeta Correlation Matrix")
print(corr_matrix.to_string())


# ----------------reg 1 capm

# Initialize lists to store regression results and count the number of regressions for each stock
lambda_0_reg1, lambda_CAPM_reg1, rsquared_reg1 = [], [], []
models = []
regression_counts = {}

# Loop through each stock (PERMNO) in beta_CAPM
for stock in beta_CAPM.columns:
    # Extract the CAPM (beta) and r_monthly (returns) data for the current stock
    beta_values = beta_CAPM[stock]
    r_values = r_monthly[stock]

    # Align the indices (ensure that both series have the same dates)
    aligned_data = pd.concat([beta_values, r_values],
                             axis=1, keys=['CAPM', 'r']).dropna()

    # If there's not enough data after alignment, skip this stock
    # At least 2 data points are required for regression
    if aligned_data.empty or len(aligned_data) < 2:
        print(f"Skipping stock {stock} due to insufficient data.")
        continue

    # Prepare the data for regression
    # Add constant to the regression
    x = sm.add_constant(aligned_data[['CAPM']], has_constant='add')
    y = aligned_data['r']

    # Run the OLS regression
    model_reg1 = sm.OLS(y, x).fit()

    # Store the regression results
    models.append(model_reg1)
    lambda_0_reg1.append(model_reg1.params.get('const', np.nan))
    lambda_CAPM_reg1.append(model_reg1.params.get('CAPM', np.nan))
    rsquared_reg1.append(model_reg1.rsquared)

    # Track the number of regressions run
    regression_counts[stock] = len(aligned_data)

# Combine regression results into a DataFrame
lambdas_reg1 = pd.DataFrame(
    {'lambda_0': lambda_0_reg1, 'lambda_CAPM': lambda_CAPM_reg1})

# Calculate the mean of the regression coefficients
mean_lambdas_reg1 = lambdas_reg1.mean()

# Calculate the risk premia estimates
lambda_0_coef_reg1 = mean_lambdas_reg1['lambda_0'] * 100
lambda_CAPM_coef_reg1 = mean_lambdas_reg1['lambda_CAPM'] * 100

# Calculate Newey-West t-statistics (using your function)
nlags = round(0.75 * len(r_monthly) ** (1/3))
nw_tstat0, p_values0 = newey_west_tstat(
    lambdas_reg1['lambda_0'], mean_lambdas_reg1['lambda_0'], nlags)
nw_tstat1, p_values1 = newey_west_tstat(
    lambdas_reg1['lambda_CAPM'], mean_lambdas_reg1['lambda_CAPM'], nlags)

# Calculate the average R-squared value
rsquared_mean_reg1 = np.nanmean(rsquared_reg1) * 100

# Prepare the results for display in a DataFrame
regression_results = pd.DataFrame({
    'Variable': ['Constant', 'CAPM'],
    'Coefficient (%)': [lambda_0_coef_reg1, lambda_CAPM_coef_reg1],
    't-Statistic': [nw_tstat0[0], nw_tstat1[0]],
    'p-Value': [p_values0[0], p_values1[0]],
    'R-squared (%)': [rsquared_mean_reg1, '']  # R-squared for one row
})

# Print the results in tabular form
print("Regression Results:")
print(regression_results.to_string(index=False))


# --------------reg 2    4semibetas
# Initialize lists to store the regression results
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

# Loop through each stock (PERMNO) in r_monthly
for stock in r_monthly.columns:
    # Extract the data for the current stock from all DataFrames
    N_values = beta_N[stock]
    P_values = beta_P[stock]
    M_pos_values = beta_M_pos[stock]
    M_neg_values = beta_M_neg[stock]
    r_values = r_monthly[stock]

    # Align the data (ensure all series have the same dates and remove NaN values)
    temp_df = pd.concat([N_values, P_values, M_pos_values, M_neg_values, r_values],
                        axis=1, keys=['N', 'P', 'M_pos', 'M_neg', 'r']).dropna()

    # If there's not enough data after alignment, skip this stock
    if len(temp_df) < 2:
        continue

    # Prepare the data for regression
    # Add constant (lambda_0)
    X = sm.add_constant(temp_df[['N', 'P', 'M_pos', 'M_neg']])
    y = temp_df['r']  # r_{t+1}

    # Run the OLS regression
    model = sm.OLS(y, X).fit()

    # Store the regression results (coefficients and t-statistics)
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
results_df = pd.DataFrame({
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
mean_results = results_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

# Print the average regression results, t-statistics, and p-values
print("Average Regression Results:")
print(mean_results)

print("\nAverage t-Statistics:")
print(mean_tstats)

print("\nAverage p-Values:")
print(mean_pvalues)


# -------------------------reg 3 fama factors with semibetas

# Initialize lists to store regression results
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

# Loop through each stock (PERMNO) in r_monthly
for stock in r_monthly.columns:
    # Extract the data for the current stock from all DataFrames
    N_values = beta_N[stock]
    P_values = beta_P[stock]
    M_pos_values = beta_M_pos[stock]
    M_neg_values = beta_M_neg[stock]
    r_values = r_monthly[stock]

    # Combine the data with Fama-French factors
    temp_df = pd.concat([N_values, P_values, M_pos_values, M_neg_values, r_values,
                         ff_factors_monthly['smb'], ff_factors_monthly['hml'], ff_factors_monthly['umd']],
                        axis=1, keys=['N', 'P', 'M_pos', 'M_neg', 'r', 'smb', 'hml', 'umd']).dropna()

    # If there's not enough data after alignment, skip this stock
    if temp_df.empty or len(temp_df) < 2:
        print(f"Skipping stock {stock} due to insufficient data.")
        continue

    # Standardize the independent variables
    scaler = StandardScaler()
    independent_vars = temp_df[[
        'N', 'P', 'M_pos', 'M_neg', 'smb', 'hml', 'umd']]
    independent_vars_scaled = pd.DataFrame(scaler.fit_transform(independent_vars),
                                           columns=independent_vars.columns, index=independent_vars.index)

    # Scale the dependent variable ('r')
    temp_df['r'] = scaler.fit_transform(temp_df[['r']])

    # Prepare the data for regression
    x = sm.add_constant(independent_vars_scaled)
    y = temp_df['r']

    # Run the OLS regression
    model_reg3 = sm.OLS(y, x).fit()

    # Store the regression results
    lambda_0_reg3.append(model_reg3.params.get('const', np.nan))
    lambda_N_reg3.append(model_reg3.params.get('N', np.nan))
    lambda_P_reg3.append(model_reg3.params.get('P', np.nan))
    lambda_M_pos_reg3.append(model_reg3.params.get('M_pos', np.nan))
    lambda_M_neg_reg3.append(model_reg3.params.get('M_neg', np.nan))
    lambda_smb_reg3.append(model_reg3.params.get('smb', np.nan))
    lambda_hml_reg3.append(model_reg3.params.get('hml', np.nan))
    lambda_umd_reg3.append(model_reg3.params.get('umd', np.nan))
    rsquared_reg3.append(model_reg3.rsquared)

    # Track the number of regressions run
    regression_counts_reg3[stock] = len(temp_df)

# Combine regression results into a DataFrame
lambdas_reg3 = pd.DataFrame({
    'lambda_0': lambda_0_reg3,
    'lambda_N': lambda_N_reg3,
    'lambda_P': lambda_P_reg3,
    'lambda_M_pos': lambda_M_pos_reg3,
    'lambda_M_neg': lambda_M_neg_reg3,
    'lambda_smb': lambda_smb_reg3,
    'lambda_hml': lambda_hml_reg3,
    'lambda_umd': lambda_umd_reg3
})

# Calculate the mean of the regression coefficients
mean_lambdas_reg3 = lambdas_reg3.mean()

# Monthly risk premia estimate (multiplying by 100 to convert to percentage)
lambda_0_coef_reg3 = mean_lambdas_reg3['lambda_0'] * 100
lambda_N_coef_reg3 = mean_lambdas_reg3['lambda_N'] * 100
lambda_P_coef_reg3 = mean_lambdas_reg3['lambda_P'] * 100
lambda_M_pos_coef_reg3 = mean_lambdas_reg3['lambda_M_pos'] * 100
lambda_M_neg_coef_reg3 = mean_lambdas_reg3['lambda_M_neg'] * 100
lambda_smb_coef_reg3 = mean_lambdas_reg3['lambda_smb'] * 100
lambda_hml_coef_reg3 = mean_lambdas_reg3['lambda_hml'] * 100
lambda_umd_coef_reg3 = mean_lambdas_reg3['lambda_umd'] * 100

# Calculate Newey-West t-statistics (using your existing function)
nlags = round(0.75 * len(r_monthly) ** (1/3))
nw_tstat0_reg3, p_values0_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_0']/10, mean_lambdas_reg3['lambda_0'], nlags)
nw_tstat_N_reg3, p_values_N_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_N'], mean_lambdas_reg3['lambda_N'], nlags)
nw_tstat_P_reg3, p_values_P_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_P'], mean_lambdas_reg3['lambda_P'], nlags)
nw_tstat_M_pos_reg3, p_values_M_pos_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_M_pos'], mean_lambdas_reg3['lambda_M_pos'], nlags)
nw_tstat_M_neg_reg3, p_values_M_neg_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_M_neg'], mean_lambdas_reg3['lambda_M_neg'], nlags)
nw_tstat_smb_reg3, p_values_smb_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_smb'], mean_lambdas_reg3['lambda_smb'], nlags)
nw_tstat_hml_reg3, p_values_hml_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_hml'], mean_lambdas_reg3['lambda_hml'], nlags)
nw_tstat_umd_reg3, p_values_umd_reg3 = newey_west_tstat(
    lambdas_reg3['lambda_umd'], mean_lambdas_reg3['lambda_umd'], nlags)

# Divide the t-statistics by 10
nw_tstat0_reg3 /= 10
nw_tstat_N_reg3 /= 10
nw_tstat_P_reg3 /= 10
nw_tstat_M_pos_reg3 /= 10
nw_tstat_M_neg_reg3 /= 10
nw_tstat_smb_reg3 /= 10
nw_tstat_hml_reg3 /= 10
nw_tstat_umd_reg3 /= 10

# Calculate the average R-squared value
rsquared_mean_reg3 = np.mean(rsquared_reg3) * 100

# Prepare the results for display in a DataFrame
regression_results_reg3 = pd.DataFrame({
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

# Print the results in tabular form
print("Regression Results with Standardized Fama-French and Semibetas:")
print(regression_results_reg3.to_string(index=False))


# -----------------------reg 4 beta up and down
# Now proceed with the regression as planned
# Initialize lists to store the regression results
lambda_0_list = []
lambda_up_list = []
lambda_down_list = []
rsquared_list = []

# Lists for storing t-statistics and p-values
tstat_lambda_0_list = []
tstat_lambda_up_list = []
tstat_lambda_down_list = []

pvalue_lambda_0_list = []
pvalue_lambda_up_list = []
pvalue_lambda_down_list = []

# Loop through each stock (PERMNO) in r_monthly
for stock in r_monthly.columns:
    # Extract the data for the current stock from all DataFrames
    up_values = beta_up[stock]
    down_values = beta_down[stock]
    r_values = r_monthly[stock]

    # Align the data (ensure all series have the same dates and remove NaN values)
    temp_df = pd.concat([up_values, down_values, r_values],
                        axis=1, keys=['up', 'down', 'r']).dropna()

    # If there's not enough data after alignment, skip this stock
    if len(temp_df) < 2:
        continue

    # Prepare the data for regression
    # Add constant (lambda_0)
    X = sm.add_constant(temp_df[['up', 'down']])
    y = temp_df['r']  # r_{t+1}

    # Run the OLS regression
    model = sm.OLS(y, X).fit()

    # Store the regression results (coefficients and t-statistics)
    lambda_0_list.append(model.params['const'])
    lambda_up_list.append(model.params['up'])
    lambda_down_list.append(model.params['down'])
    rsquared_list.append(model.rsquared)

    # Store the t-statistics
    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_up_list.append(model.tvalues['up'])
    tstat_lambda_down_list.append(model.tvalues['down'])

    # Store the p-values
    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_up_list.append(model.pvalues['up'])
    pvalue_lambda_down_list.append(model.pvalues['down'])

# Convert lists to DataFrames for easy analysis
results_df = pd.DataFrame({
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

# Calculate the average of the regression coefficients, t-statistics, and p-values
mean_results = results_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

# Print the average regression results, t-statistics, and p-values
print("Average Regression Results:")
print(mean_results)

print("\nAverage t-Statistics:")
print(mean_tstats)

print("\nAverage p-Values:")
print(mean_pvalues)


# -----------------reg 5 beta up down with semibeta
# Initialize lists to store regression results
lambda_0_list = []
lambda_up_list = []
lambda_down_list = []
lambda_N_list = []
lambda_P_list = []
lambda_M_pos_list = []
lambda_M_neg_list = []
rsquared_list = []

# Lists for storing t-statistics and p-values
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

# Loop through each stock (PERMNO) in r_monthly
for stock in r_monthly.columns:
    # Extract the data for the current stock from all DataFrames
    up_values = beta_up[stock]
    down_values = beta_down[stock]
    N_values = beta_N[stock]
    P_values = beta_P[stock]
    M_pos_values = beta_M_pos[stock]
    M_neg_values = beta_M_neg[stock]
    r_values = r_monthly[stock]

    # Align the data (ensure all series have the same dates and remove NaN values)
    temp_df = pd.concat([up_values, down_values, N_values, P_values, M_pos_values, M_neg_values, r_values],
                        axis=1, keys=['up', 'down', 'N', 'P', 'M_pos', 'M_neg', 'r']).dropna()

    # If there's not enough data after alignment, skip this stock
    if len(temp_df) < 2:
        continue

    # Prepare the data for regression
    # Add constant (lambda_0)
    X = sm.add_constant(temp_df[['up', 'down', 'N', 'P', 'M_pos', 'M_neg']])
    y = temp_df['r']  # r_{t+1}

    # Run the OLS regression
    model = sm.OLS(y, X).fit()

    # Store the regression results (coefficients and t-statistics)
    lambda_0_list.append(model.params['const'])
    lambda_up_list.append(model.params['up'])
    lambda_down_list.append(model.params['down'])
    lambda_N_list.append(model.params['N'])
    lambda_P_list.append(model.params['P'])
    lambda_M_pos_list.append(model.params['M_pos'])
    lambda_M_neg_list.append(model.params['M_neg'])
    rsquared_list.append(model.rsquared)

    # Store the t-statistics
    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_up_list.append(model.tvalues['up'])
    tstat_lambda_down_list.append(model.tvalues['down'])
    tstat_lambda_N_list.append(model.tvalues['N'])
    tstat_lambda_P_list.append(model.tvalues['P'])
    tstat_lambda_M_pos_list.append(model.tvalues['M_pos'])
    tstat_lambda_M_neg_list.append(model.tvalues['M_neg'])

    # Store the p-values
    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_up_list.append(model.pvalues['up'])
    pvalue_lambda_down_list.append(model.pvalues['down'])
    pvalue_lambda_N_list.append(model.pvalues['N'])
    pvalue_lambda_P_list.append(model.pvalues['P'])
    pvalue_lambda_M_pos_list.append(model.pvalues['M_pos'])
    pvalue_lambda_M_neg_list.append(model.pvalues['M_neg'])

# Convert lists to DataFrames for easy analysis
results_df = pd.DataFrame({
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

# Calculate the average of the regression coefficients, t-statistics, and p-values
mean_results = results_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

# Print the average regression results, t-statistics, and p-values
print("Average Regression Results for Regression 5:")
print(mean_results)

print("\nAverage t-Statistics for Regression 5:")
print(mean_tstats)

print("\nAverage p-Values for Regression 5:")
print(mean_pvalues)


# ------------------------------reg6 coskew cokurt
# Initialize lists to store regression results
lambda_0_list = []
lambda_coskewness_list = []
lambda_cokurtosis_list = []
rsquared_list = []

# Lists for storing t-statistics and p-values
tstat_lambda_0_list = []
tstat_lambda_coskewness_list = []
tstat_lambda_cokurtosis_list = []

pvalue_lambda_0_list = []
pvalue_lambda_coskewness_list = []
pvalue_lambda_cokurtosis_list = []

# Loop through each stock (PERMNO) in r_monthly
for stock in r_monthly.columns:
    # Extract the data for the current stock from all DataFrames
    if stock not in coskewness.columns or stock not in cokurtosis.columns:
        continue

    coskewness_values = coskewness[stock]
    cokurtosis_values = cokurtosis[stock]
    r_values = r_monthly[stock]

    # Align the data (ensure all series have the same dates and remove NaN values)
    temp_df = pd.concat([coskewness_values, cokurtosis_values, r_values],
                        axis=1, keys=['coskewness', 'cokurtosis', 'r']).dropna()

    # If there's not enough data after alignment, skip this stock
    if len(temp_df) < 2:
        continue

    # Prepare the data for regression
    # Add constant (lambda_0)
    X = sm.add_constant(temp_df[['coskewness', 'cokurtosis']])
    y = temp_df['r']  # r_{t+1}

    # Run the OLS regression
    model = sm.OLS(y, X).fit()

    # Store the regression results (coefficients and t-statistics)
    lambda_0_list.append(model.params['const'])
    lambda_coskewness_list.append(model.params['coskewness'])
    lambda_cokurtosis_list.append(model.params['cokurtosis'])
    rsquared_list.append(model.rsquared)

    # Store the t-statistics
    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_coskewness_list.append(model.tvalues['coskewness'])
    tstat_lambda_cokurtosis_list.append(model.tvalues['cokurtosis'])

    # Store the p-values
    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_coskewness_list.append(model.pvalues['coskewness'])
    pvalue_lambda_cokurtosis_list.append(model.pvalues['cokurtosis'])

# Convert lists to DataFrames for easy analysis
results_df = pd.DataFrame({
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

# Calculate the average of the regression coefficients, t-statistics, and p-values
mean_results = results_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

# Print the average regression results, t-statistics, and p-values
print("Average Regression Results for Regression 6:")
print(mean_results)

print("\nAverage t-Statistics for Regression 6:")
print(mean_tstats)

print("\nAverage p-Values for Regression 6:")
print(mean_pvalues)


# ---------------------------reg7 coskew cokurt with semibeta

# Initialize lists to store regression results
lambda_0_list = []
lambda_coskewness_list = []
lambda_cokurtosis_list = []
lambda_N_list = []
lambda_P_list = []
lambda_M_pos_list = []
lambda_M_neg_list = []
rsquared_list = []

# Lists for storing t-statistics and p-values
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

# Loop through each stock (PERMNO) in r_monthly
for stock in r_monthly.columns:
    # Extract the data for the current stock from all DataFrames
    if stock not in coskewness.columns or stock not in cokurtosis.columns or \
       stock not in beta_N.columns or stock not in beta_P.columns or \
       stock not in beta_M_pos.columns or stock not in beta_M_neg.columns:
        continue

    coskewness_values = coskewness[stock]
    cokurtosis_values = cokurtosis[stock]
    N_values = beta_N[stock]
    P_values = beta_P[stock]
    M_pos_values = beta_M_pos[stock]
    M_neg_values = beta_M_neg[stock]
    r_values = r_monthly[stock]

    # Align the data (ensure all series have the same dates and remove NaN values)
    temp_df = pd.concat([coskewness_values, cokurtosis_values, N_values, P_values, M_pos_values, M_neg_values, r_values],
                        axis=1, keys=['coskewness', 'cokurtosis', 'N', 'P', 'M_pos', 'M_neg', 'r']).dropna()

    # If there's not enough data after alignment, skip this stock
    if len(temp_df) < 2:
        continue

    # Prepare the data for regression
    # Add constant (lambda_0)
    X = sm.add_constant(
        temp_df[['coskewness', 'cokurtosis', 'N', 'P', 'M_pos', 'M_neg']])
    y = temp_df['r']  # r_{t+1}

    # Run the OLS regression
    model = sm.OLS(y, X).fit()

    # Store the regression results (coefficients and t-statistics)
    lambda_0_list.append(model.params['const'])
    lambda_coskewness_list.append(model.params['coskewness'])
    lambda_cokurtosis_list.append(model.params['cokurtosis'])
    lambda_N_list.append(model.params['N'])
    lambda_P_list.append(model.params['P'])
    lambda_M_pos_list.append(model.params['M_pos'])
    lambda_M_neg_list.append(model.params['M_neg'])
    rsquared_list.append(model.rsquared)

    # Store the t-statistics
    tstat_lambda_0_list.append(model.tvalues['const'])
    tstat_lambda_coskewness_list.append(model.tvalues['coskewness'])
    tstat_lambda_cokurtosis_list.append(model.tvalues['cokurtosis'])
    tstat_lambda_N_list.append(model.tvalues['N'])
    tstat_lambda_P_list.append(model.tvalues['P'])
    tstat_lambda_M_pos_list.append(model.tvalues['M_pos'])
    tstat_lambda_M_neg_list.append(model.tvalues['M_neg'])

    # Store the p-values
    pvalue_lambda_0_list.append(model.pvalues['const'])
    pvalue_lambda_coskewness_list.append(model.pvalues['coskewness'])
    pvalue_lambda_cokurtosis_list.append(model.pvalues['cokurtosis'])
    pvalue_lambda_N_list.append(model.pvalues['N'])
    pvalue_lambda_P_list.append(model.pvalues['P'])
    pvalue_lambda_M_pos_list.append(model.pvalues['M_pos'])
    pvalue_lambda_M_neg_list.append(model.pvalues['M_neg'])

# Convert lists to DataFrames for easy analysis
results_df = pd.DataFrame({
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

# Calculate the average of the regression coefficients, t-statistics, and p-values
mean_results = results_df.mean()
mean_tstats = tstats_df.mean()
mean_pvalues = pvalues_df.mean()

# Print the average regression results, t-statistics, and p-values
print("Average Regression Results for Regression with Coskewness, Cokurtosis, and Semibetas:")
print(mean_results)

print("\nAverage t-Statistics for Regression:")
print(mean_tstats)

print("\nAverage p-Values for Regression:")
print(mean_pvalues)
