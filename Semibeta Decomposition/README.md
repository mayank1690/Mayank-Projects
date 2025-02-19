# **Exploring the Predictive Power of Semibetas in Asset Pricing**  

Note: A detailed description of the project is present in the PDF file attached, same has been breifly described below

## **Introduction**  
This project investigates the predictive power of semibetas in explaining asset returns, comparing them to traditional asset pricing models such as the Capital Asset Pricing Model (CAPM) and the Fama-French factors. The study decomposes market beta into four distinct semibetas and tests them in asset pricing models.  

The results reveal that semibetas, particularly those related to negative market movements, offer a better explanation of asset returns than conventional models. This suggests that incorporating semibetas into asset pricing models can significantly enhance their predictive power, offering valuable insights for investors and researchers in portfolio management.  

A more detailed description of the project is available in the accompanying PDF document.  

### **Limitations of CAPM**  
The CAPM has several theoretical limitations, including:  
- Unrealistic assumptions such as homogeneous expectations, efficient markets, and a single-period investment horizon.  
- Poor explanatory power for cross-sectional stock returns.  
- Failure to account for downside risk and asymmetric market behavior.  

Bollerslev et al. (2022) introduced a method to decompose market beta into four semibetas:  
- **βN (Beta Negative)** – Sensitivity to negative market returns.  
- **βP (Beta Positive)** – Sensitivity to positive market returns.  
- **βM+ (Mixed Beta Positive)** – When stock returns are negative and market returns are positive.  
- **βM− (Mixed Beta Negative)** – When stock returns are positive and market returns are negative.  

These semibetas provide a more refined understanding of stock risk exposure than traditional beta.  

## **Methodology**  

### **Data Source & Stock Selection**  
The primary data source is the **CRSP (Center for Research in Security Prices) database**, covering **January 1963 to December 2019**. The dataset includes:  
- **Stock Identifiers** (PERMNO, PERMCO)  
- **Price & Volume Data** (PRC, VOL, RET, SHROUT)  
- **Fama-French Factors** (MKT, SMB, HML, UMD)  

### **Data Cleaning**  
To ensure a robust dataset, the following preprocessing steps were applied:  
- **Standardization & Formatting:** Dates were converted to a uniform format, and numerical fields were validated.  
- **Filtering:** Penny stocks (price < $5) and illiquid stocks were removed.  
- **Handling Missing Data:** Forward and backward filling was applied to minimize missing values.  
- **Outlier Treatment:** Winsorization was applied at the **1st and 99th percentiles** to reduce extreme outliers.  
- **Final Dataset:** The cleaned dataset contained **194 stocks** spanning **56 years**.  

### **Market Return Calculation**  
A **value-weighted market index** was created based on daily market capitalization. Market returns were computed as a weighted average of stock returns.  

### **Semibeta Computation**  
Stock returns were separated into positive and negative components, allowing for semibeta calculations:  
- **βN and βP** capture movements when the stock and market move in the same direction.  
- **βM+ and βM−** capture cases where stock and market move in opposite directions.  

These semibetas provide a nuanced view of stock sensitivity under different market conditions.  

### **Comparison Models**  
To benchmark semibetas, the study also incorporated:  
- **Fama-French Factors:** Market premium (MKT), size (SMB), value (HML), and momentum (UMD).  
- **Higher-Order Moments:** **Coskewness and Cokurtosis** to measure asymmetric risks.  
- **Upside & Downside Betas:** To assess stock response to positive vs. negative market movements.  

### **Regression Models**  
Linear regression with the **Newey-West correction** was applied to estimate risk premia, ensuring robust standard errors. The study performed:  
1. **CAPM Regression:** Traditional market beta vs. stock returns.  
2. **Semibeta Regressions:** Testing semibetas' impact on stock returns.  
3. **Fama-French & Semibeta Regressions:** Integrating semibetas with traditional factors.  
4. **Higher-Order Moment Regressions:** Adding coskewness and cokurtosis.  
5. **Upside & Downside Beta Regressions:** Comparing semibetas to upside/downside betas.  
6. **Combined Regressions:** Evaluating semibetas alongside Fama-French factors and higher-order moments.  

---

## **Results & Key Findings**  

### **Summary Statistics**  
- **Concordant Semibetas (βN, βP) were larger than discordant semibetas (βM+, βM−)**, indicating that stocks tend to move in the same direction as the market more often than not.  
- **Traditional beta had a mean of 0.8182**, showing that stocks exhibit a less-than-proportional response to market movements.  

### **Regression Results**  
#### **Monthly Return Analysis**  
- The **traditional CAPM beta explains only 10.7% of return variations**.  
- **Semibetas increased explanatory power to 61.4%**, proving their significance in asset pricing.  
- **βN and βP had the highest statistical significance**, confirming that downside risk is a key factor in pricing.  
- **Fama-French factors (SMB, HML, UMD) did not add significant explanatory power** beyond semibetas.  

#### **Weekly Return Analysis**  
- Market beta’s significance **declined in shorter timeframes** (R² dropped from 10.7% in monthly to 6.25% in weekly).  
- **Semibetas remained highly significant, increasing R² to 58.3%.**  
- **Upside and Downside betas had lower explanatory power compared to semibetas.**  

#### **Daily Return Analysis**  
- The **predictive power of semibetas declined at the daily frequency** but remained statistically significant.  
- **Upside & Downside betas were more influential at the daily level**, confirming the role of short-term market movements.  

### **Higher-Order Moments & Adjusted R² Comparison**  
- **Coskewness and cokurtosis alone were not significant predictors of returns.**  
- When combined with semibetas, **coskewness became significant, improving R² to 67.08%.**  
- **Adjusted R² Comparisons:**  
  - CAPM: **3.68%**  
  - Semibetas: **52.36%**  
  - Semibetas + Fama-French: **56.98%**  
  - Semibetas + Coskewness/Cokurtosis: **62.23%**  

---

## **Conclusion**  
This study found that semibetas provide a **more detailed and effective explanation** of asset returns compared to traditional market beta. The key takeaways are:  
- **Downside risk plays a crucial role in asset pricing.**  
- **Semibetas significantly outperform CAPM and Fama-French models in explaining return variations.**  
- **Investors can leverage semibetas to assess and manage downside risks more effectively.**  
- **Portfolio managers can use semibetas to improve risk estimation, especially during market downturns.**  

By incorporating semibetas, researchers and practitioners can enhance asset pricing models to account for asymmetric risk exposure, leading to better risk management and portfolio optimization.  
