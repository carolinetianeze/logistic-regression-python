#%% Installation of required libraries
# pip install pandas numpy seaborn matplotlib statsmodels kagglehub openpyxl

#%% Import of required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit
import kagglehub
import openpyxl
import os

#%% *** 1. LOADING DATA ***
# A. Downloading the latest version of the dataset
path = kagglehub.dataset_download('debjeetdas/babies-birth-weight')

# B. Identifying the file in the download directory
files = os.listdir(path)
file_path = os.path.join(path, files[0])

# C. Reading the CSV file
df = pd.read_csv(file_path)
print('BWT Data Preview:', df.head())

# D. Create Figures and Tables directories if they don't exist
if not os.path.exists('low_birthweight/Figures'):
    os.makedirs('low_birthweight/Figures')

if not os.path.exists('low_birthweight/Tables'):
    os.makedirs('low_birthweight/Tables')

#%% *** 2. DATA CLEANING & TRANSFORMATION ***

# A. Dropping rows with missing values in key predictors
df_clean = df.dropna(subset=['bwt', 'gestation', 'age', 'height', 'weight', 'smoke']).copy()

# B. Defining Low Birth Weight (LBW)
# In this dataset, bwt is in ounces (oz). 2500g is approx. 88oz.
df_clean['low_bwt'] = (df_clean['bwt'] < 88).astype(int)

# C. BMI Calculation: 703 * weight_lb / (height_in ** 2)
df_clean['bmi'] = 703.0 * df_clean['weight'] / (df_clean['height'] ** 2)

# D. Grouping BMI categories to improve statistical power for the logistic model
def bmi_category(b):
    if b < 18.5:
        return 'Underweight'
    elif b < 25:
        return 'Normal'
    else:
        return 'Overweight_Obese'

df_clean['bmi_cat'] = df_clean['bmi'].apply(bmi_category)

#%% *** 3. DESCRIPTIVE ANALYSIS ***
print('\n--- Low Birth Weight Frequency ---')
print(df_clean['low_bwt'].value_counts(normalize=True) * 100)

# A. Correlation Heatmap (Continuous variables)
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean[['bwt', 'gestation', 'age', 'bmi']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix: Clinical Factors')

# B. Visualization: LBW Proportion by Smoking Status
plt.figure(figsize=(8, 6))
sns.barplot(x='smoke', y='low_bwt', data=df_clean, palette='viridis')
plt.title('Proportion of Low Birth Weight: Smokers (1) vs Non-Smokers (0)')
plt.ylabel('LBW Probability')
plt.savefig('Low_Birthweight/Figures/low_bwt_by_smoke.png')

# C. Visualization: LBW by Gestation Duration
plt.figure(figsize=(10, 6))
sns.boxplot(x='low_bwt', y='gestation', data=df_clean)
plt.title('Gestation Period vs Birth Weight Category')
plt.xlabel('Low Birth Weight (0=No, 1=Yes)')
plt.ylabel('Gestation (Days)')
plt.savefig('Low_Birthweight/Figures/gestation_boxplot.png')
plt.show()

#%% *** 4. LOGISTIC REGRESSION MODEL ***
# Hypothesis: Smoking, Gestation, and BMI are significant predictors of Low Birth Weight
# We use 'Normal' BMI as the reference category

# A. Logistic Regression
formula = "low_bwt ~ C(smoke) + gestation + age + C(bmi_cat, Treatment(reference='Normal'))"
model_logit = logit(formula, data=df_clean).fit()

print('\n--- Logistic Regression Summary ---')
print(model_logit.summary())

# B. Likelihood Ratio Test (LRT)
# Tests if the full model is better than an empty model (output in summary)
print(f"Log-Likelihood Ratio p-value: {model_logit.llr_pvalue:.4f}")

if model_logit.llr_pvalue < 0.05:
    print("Decision: The model is statistically significant (at least one predictor is relevant).")
else:
    print("Decision: The model is not statistically significant.")

# C. Calculating Odds Ratios (OR)
''' OR > 1 indicates increased odds of LBW
OR < 1 indicates protective effect'''

# Getting raw params and CI
params = model_logit.params
conf = model_logit.conf_int()

# Combining them into one DataFrame
results_df = pd.DataFrame({
    'Odds Ratio': params,
    'Lower CI': conf[0],
    'Upper CI': conf[1]
})

# Exponentiating to get actual Odds Ratios
results_df = np.exp(results_df)

# Cleaning up the index for Excel
output_df = results_df.reset_index()
output_df.columns = ['Variable', 'Odds Ratio', 'Lower CI', 'Upper CI']

# Saving
output_df.to_excel('Low_Birthweight/Tables/logit_results.xlsx', index=False)
print('File saved successfully!')

#%% 5. FOREST PLOT

# A. Filtering out Intercept and renaming variables for the plot
plot_df = output_df[output_df['Variable'] != 'Intercept'].copy()

name_mapping = {
    "C(smoke)[T.1.0]": "Smoker",
    "C(bmi_cat, Treatment(reference='Normal'))[T.Underweight]": "Underweight",
    "C(bmi_cat, Treatment(reference='Normal'))[T.Overweight_Obese]": "Overweight/Obese",
    "gestation": "Gestation Days",
    "age": "Mother's Age"
}

plot_df['Variable'] = plot_df['Variable'].replace(name_mapping)
plot_df = plot_df.sort_values('Odds Ratio', ascending=True)

# B. Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(x=plot_df['Odds Ratio'], y=plot_df['Variable'],
             xerr=[plot_df['Odds Ratio'] - plot_df['Lower CI'],
                   plot_df['Upper CI'] - plot_df['Odds Ratio']],
             fmt='o', color='black', ecolor='darkred', capsize=5, markersize=8)
plt.axvline(x=1, color='blue', linestyle='--', alpha=0.6) # Line of no effect
plt.title('Adjusted OR for Low Birth Weight', fontsize=14)
plt.xlabel('Odds Ratio (95% CI)', fontsize=12)
plt.ylabel('Predictors', fontsize=12)
plt.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("Low_Birthweight/Figures/forest_plot_lbw_cleaned.png")
plt.show()

#%% *** 6. GOODNESS-OF-FIT TESTS ***

# A. Pseudo R-squared (McFadden's)
print(f"McFadden's Pseudo R-squared: {model_logit.prsquared:.4f}")

# B. Pearson Chi-Square Goodness-of-Fit
# p-values > 0.05 indicate a good fit
pearson_chi2 = np.sum(model_logit.resid_pearson**2)
df_resid = model_logit.df_resid
p_value_chi2 = 1 - stats.chi2.cdf(pearson_chi2, df_resid)

print(f"Pearson Chi-Square: {pearson_chi2:.4f}")
print(f"Degrees of Freedom: {df_resid}")
print(f"Pearson Goodness-of-Fit p-value: {p_value_chi2:.4f}")

# C. Deviance Goodness-of-Fit
# Deviance = -2 * Log-Likelihood
deviance = -2 * model_logit.llf
p_value_deviance = 1 - stats.chi2.cdf(deviance, df_resid)

print(f"Calculated Deviance: {deviance:.4f}")
print(f"Deviance p-value: {p_value_deviance:.4f}")

# Overall Conclusion
if p_value_chi2 > 0.05 and p_value_deviance > 0.05:
    print("\nInterpretation: Both Pearson and Deviance tests suggest the model fits the data well.")
else:
    print("\nInterpretation: There may be some lack of fit; check for non-linearities or interactions.")
