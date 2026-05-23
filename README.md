# Infant Birth Weight Analysis: Biological and Lifestyle Determinants
This project conducts a comprehensive statistical analysis of the factors influencing newborn birth weight (in ounces) and the likelihood of **Low Birth Weight (LBW)**. Using a dataset of 1,174 cleaned records, we transition from linear regression to a **Logistic Regression** framework to identify the odds of clinical risk.

## Key Research Findings
### 1. The "Smoking Penalty" (Odds Ratio)
Smoking is confirmed as the most critical modifiable risk factor.
* **Increased Risk:** Mothers who smoke are **3.37 times more likely** (p < 0.001) to have a Low Birth Weight infant compared to non-smokers.
* **Consistency:** This effect remains highly significant even when controlling for gestation, maternal age, and BMI.

### 2. Gestation: The Primary Biological Predictor
Gestation length is the strongest protective factor against LBW.
* **Protective Effect:** For every additional day of gestation, the odds of LBW decrease by approximately **7.3%** (p < 0.001).
* **Significance:** The relationship is robust, highlighting the importance of full-term pregnancy.

### 3. Maternal BMI and Age
* **BMI Categories:** While "Underweight" and "Overweight/Obese" categories showed positive coefficients (suggesting higher odds of LBW relative to "Normal" BMI), they did not reach the  significance threshold in this logistic model (p > 0.05).
* **Maternal Age:** Age showed a slight positive correlation with LBW odds, but was not statistically significant at the 5% level.

## Model Performance & Reliability
To ensure the scientific validity of the findings, we performed several goodness-of-fit tests:

| Test | Result | Interpretation |
| --- | --- | --- |
| **McFadden’s Pseudo** | 0.2549 | Strong explanatory power for a logistic model. |
| **LLR p-value** | < 0.0001 | The full model is better than an empty model. |
| **Pearson Chi-Square** |  | Suggests the model fits the data well. |
| **Deviance** |  | No evidence of significant lack-of-fit. |

## Feature Engineering & Methodology
* **BMI Calculation:** Derived using the standard formula:
$$\text{BMI} = 703 \times \frac{\text{weight (lb)}}{\text{height (in)}^2}$$
* **Categorization:** BMI was binned into `Underweight`, `Normal` (Reference), and `Overweight_Obese` to improve statistical power.
* **Low Birth Weight (LBW) Classification:** Infants weighing less than **88 oz** (~2500g) were categorized as LBW.

## Project Structure
* `lbw_logistic_regression.py`: Full pipeline (Data cleaning, Logistic Regression, Goodness-of-Fit tests).
* **Figures/**:
* `low_bwt_by_smoke.png`: Bar chart of LBW probability by smoking status.
* `gestation_boxplot.png`: Distribution of gestation days across birth weight categories.
* `forest_plot_lbw_cleaned.png`: Visualization of Odds Ratios and 95% Confidence Intervals.
* **Tables/**:
* `logit_results.xlsx`: Exported Odds Ratios and confidence intervals for reporting.

## Public Health Implications
The transition to a logistic model reinforces that smoking is not just a weight reducer, but a primary driver of clinical "Low Birth Weight" status. Prenatal interventions focusing on smoking cessation and extending gestation remain the two most effective pathways for improving neonatal health outcomes.

---
**Developed by Caroline** – *Pharmacoepidemiologist & RWE Analytics Consultant*
