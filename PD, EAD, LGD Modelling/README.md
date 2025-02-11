# **Credit Risk Modeling: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD)**  

## **Project Overview**  
This project develops models for **credit risk assessment**, focusing on **Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD)**. The models are trained using **loan data** and aim to improve financial institutions' ability to estimate credit risk exposure.  

The project follows a structured approach:  
1. **Data Preprocessing**: Cleaning and preparing loan data.  
2. **PD Model**: Logistic regression to estimate the probability of a borrower defaulting.  
3. **EAD & LGD Models**: Regression models to predict the exposure at the time of default and the loss after default.  
4. **Expected Loss Calculation**: Combining PD, LGD, and EAD to estimate expected credit loss.  
5. **Performance Metrics**: Using **Weight of Evidence (WoE), Information Value (IV), Gini Index, and Kolmogorov-Smirnov (KS) Statistic** for feature selection and model evaluation.  

---

## **1. Data Preprocessing**  
**File:** `1. Data Preprocessing for PD.py`  

### **Data Cleaning & Feature Engineering**  
- **Missing Data Handling**:  
  - `mths_since_last_delinq` and `mths_since_last_record` are imputed with `0`.  
  - Missing categories are flagged with separate dummy variables.  
- **Categorical Variable Encoding**:  
  - **Weight of Evidence (WoE)** transformation is applied to categorical variables to improve model interpretability.  
  - **Dummy Variables** are created while ensuring no multicollinearity by removing reference categories.  

### **Weight of Evidence (WoE) & Information Value (IV)**
- **WoE** is used to transform categorical variables based on their relationship with default risk.  
- **Information Value (IV)** is calculated to determine the predictive power of each variable:  
  - **IV > 0.3** → Strong predictor  
  - **IV between 0.1 and 0.3** → Medium predictor  
  - **IV < 0.1** → Weak predictor (Removed from the model)  

These steps help in selecting **only the most relevant variables** for the PD model.  

---

## **2. Probability of Default (PD) Model**  
**File:** `2. PD model.py`  

### **Modeling Approach**  
A **logistic regression model** is used to estimate the probability that a borrower will default.  

### **Steps:**  
1. **Feature Selection Using WoE & IV**  
   - Variables with high **Information Value (IV)** are retained.  
   - Reference categories are dropped to avoid multicollinearity.  
2. **Logistic Regression**  
   - **Fitting the Model**: Trains a logistic regression model using loan attributes.  
   - **Feature Importance**: Extracts coefficients and computes **p-values** to evaluate statistical significance.  
   - **Final Model Selection**: Only significant variables are retained.  
3. **Performance Evaluation**  
   - **Kolmogorov-Smirnov (KS) Statistic**: Measures the separation between good and bad borrowers.  
   - **Gini Index**: Measures the discriminatory power of the model.  
   - **ROC Curve & AUC**: Evaluates the predictive power of the model.  
   - **Confusion Matrix**: Assesses model accuracy.  

### **Kolmogorov-Smirnov (KS) Statistic**  
- The **KS statistic** quantifies how well the model separates defaulters from non-defaulters.  
- A **higher KS value (closer to 1)** indicates better separation between the two groups.  

### **Gini Index**  
- The **Gini Index** is derived from the ROC curve:  
  \[
  Gini = 2 	imes AUC - 1
  \]
- A **higher Gini Index (closer to 1)** suggests a better-performing model.  

---

## **3. Loss Given Default (LGD) and Exposure at Default (EAD) Models**  
**File:** `3. EAD-LGD model and EL.py`  

### **Loss Given Default (LGD)**  
**Definition**: LGD measures the percentage of loan exposure lost after default.  

#### **Modeling Steps:**  
1. **Defining LGD**:  
   - Recovery rate is calculated as:  
     \[
     	ext{Recovery Rate} = rac{	ext{Recoveries}}{	ext{Funded Amount}}
     \]
   - LGD is then:  
     \[
     	ext{LGD} = 1 - 	ext{Recovery Rate}
     \]  
2. **Two-Stage LGD Model:**  
   - **Stage 1 (Logistic Regression)**: Predicts whether the recovery rate is `0` or `>0`.  
   - **Stage 2 (Linear Regression)**: Estimates the recovery rate if it's greater than `0`.  
3. **Performance Evaluation:**  
   - **Kolmogorov-Smirnov (KS) Statistic** for Stage 1 (classification).  
   - **R² and RMSE** for Stage 2 (regression).  

---

### **Exposure at Default (EAD)**  
**Definition**: EAD estimates the total exposure at the time of default.  

#### **Modeling Steps:**  
1. **Defining EAD**:  
   - Calculated using **Credit Conversion Factor (CCF)**:  
     \[
     	ext{CCF} = rac{	ext{Funded Amount} - 	ext{Total Recovered Principal}}{	ext{Funded Amount}}
     \]
   - EAD is then:  
     \[
     	ext{EAD} = 	ext{Funded Amount} 	imes 	ext{CCF}
     \]  
2. **Linear Regression Model:**  
   - Predicts **CCF** using borrower and loan characteristics.  
   - Evaluates performance using **R² and RMSE**.  

---

## **4. Expected Loss Calculation**  
Using the trained models, **Expected Loss (EL)** is computed as:  
\[
EL = PD 	imes LGD 	imes EAD
\]
This provides an estimate of potential credit losses, helping financial institutions set aside adequate capital reserves.  

---

## **Key Findings**  
- The **PD model** shows strong predictive power with **significant explanatory variables** such as **loan grade, interest rates, and credit history**.  
- **WoE and IV were effective in feature selection**, improving model interpretability.  
- **KS Statistic and Gini Index confirm that the PD model has good discriminatory power.**  
- **LGD follows a two-stage modeling approach**, improving accuracy in estimating recovery rates.  
- **EAD predictions using CCF provide a refined estimate of credit exposure at default.**  

---

## **Conclusion**  
This project provides a **robust framework for credit risk modeling** using **PD, LGD, and EAD** models. By incorporating **advanced risk metrics (WoE, IV, KS Statistic, and Gini Index)**, financial institutions can make **informed lending decisions** and **manage risk exposure effectively**.  
