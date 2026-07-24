import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Set seed for reproducibility
np.random.seed(42)

# ==========================================
# STEP 1: EXPERIMENT DESIGN & POWER ANALYSIS
# ==========================================
baseline_cr = 0.10  # 10% conversion rate
expected_cr = 0.115 # 11.5% conversion rate (15% relative lift)
mde = expected_cr - baseline_cr
alpha = 0.05
power = 0.80

# Estimate required sample size per group
std_dev = np.sqrt(baseline_cr * (1 - baseline_cr))
effect_size = mde / std_dev
req_sample_size = int(tt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1))

print(f"--- 1. POWER ANALYSIS ---")
print(f"Required sample size per group: {req_sample_size:,}\n")

# ==========================================
# STEP 2: SIMULATE REALISTIC EXPERIMENT DATA
# ==========================================
n_samples = 30000  # 15,000 per variant

user_ids = [f"USR_{10000 + i}" for i in range(n_samples)]
groups = np.random.choice(['control', 'variant'], size=n_samples, p=[0.50, 0.50])
devices = np.random.choice(['Mobile', 'Desktop'], size=n_samples, p=[0.65, 0.35])
dates = pd.date_range(start="2026-03-01", periods=14).repeat(n_samples // 14 + 1)[:n_samples]

# Generate conversion status with a true lift in variant
converted = []
for group in groups:
    p = 0.116 if group == 'variant' else 0.100  # Variant converts at 11.6%
    converted.append(np.random.binomial(1, p))

# Generate Order Value (Log-normal distribution for non-zero orders)
order_values = []
for conv in converted:
    if conv == 1:
        # Average order value around $65 with long right tail
        val = np.round(np.random.lognormal(mean=4.15, sigma=0.4), 2)
        order_values.append(val)
    else:
        order_values.append(0.0)

# Build DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'timestamp': dates,
    'group': groups,
    'device': devices,
    'converted': converted,
    'order_value': order_values
})

# Save to CSV for your SQL/Tableau import
df.to_csv("ab_test_checkout_data.csv", index=False)
print("--- 2. DATASET GENERATION ---")
print("Dataset created: 'ab_test_checkout_data.csv' (30,000 records)")
print(df.head(), "\n")

# ==========================================
# STEP 3: DATA VALIDATION & SRM CHECK
# ==========================================
print("--- 3. DATA VALIDATION ---")

# Sample Ratio Mismatch (SRM) test
obs_control = (df['group'] == 'control').sum()
obs_variant = (df['group'] == 'variant').sum()
chi2, p_srm = stats.chisquare(f_obs=[obs_control, obs_variant], f_exp=[n_samples/2, n_samples/2])

print(f"Control Count: {obs_control:,} | Variant Count: {obs_variant:,}")
print(f"SRM Chi-Square p-value: {p_srm:.4f}")
if p_srm < 0.05:
    print("WARNING: Sample Ratio Mismatch detected! Check randomization logic.\n")
else:
    print("SUCCESS: No Sample Ratio Mismatch detected. Traffic split is valid.\n")

# ==========================================
# STEP 4: HYPOTHESIS TESTING
# ==========================================
print("--- 4. STATISTICAL ANALYSIS ---")

# Primary Metric: Conversion Rate (Z-Test)
ctrl_conv = df[df['group'] == 'control']['converted']
var_conv = df[df['group'] == 'variant']['converted']

cr_ctrl = ctrl_conv.mean()
cr_var = var_conv.mean()
abs_lift = cr_var - cr_ctrl
rel_lift = abs_lift / cr_ctrl

successes = [var_conv.sum(), ctrl_conv.sum()]
nobs = [len(var_conv), len(ctrl_conv)]

z_stat, p_val = proportions_ztest(successes, nobs, alternative='larger')

# Confidence Intervals for lift
ci_ctrl = proportion_confint(ctrl_conv.sum(), len(ctrl_conv), alpha=0.05)
ci_var = proportion_confint(var_conv.sum(), len(var_conv), alpha=0.05)

print(f"Control Conversion Rate: {cr_ctrl:.2%} (95% CI: {ci_ctrl[0]:.2%} - {ci_ctrl[1]:.2%})")
print(f"Variant Conversion Rate: {cr_var:.2%} (95% CI: {ci_var[0]:.2%} - {ci_var[1]:.2%})")
print(f"Relative Lift: +{rel_lift:.2%}")
print(f"Z-Statistic: {z_stat:.4f} | P-Value: {p_val:.4e}")

# Secondary Metric: Average Order Value (AOV) via Welch's t-test
ctrl_aov = df[(df['group'] == 'control') & (df['converted'] == 1)]['order_value']
var_aov = df[(df['group'] == 'variant') & (df['converted'] == 1)]['order_value']

t_stat, p_val_aov = stats.ttest_ind(var_aov, ctrl_aov, equal_var=False)

print(f"\nControl AOV: ${ctrl_aov.mean():.2f}")
print(f"Variant AOV: ${var_aov.mean():.2f}")
print(f"AOV Difference P-Value: {p_val_aov:.4f}")

# ==========================================
# STEP 5: BUSINESS IMPACT & RECOMMENDATION
# ==========================================
print("\n--- 5. BUSINESS IMPACT & RECOMMENDATION ---")

monthly_users = 100000  # Assume 100k monthly checkout visitors
avg_order_val = df[df['converted'] == 1]['order_value'].mean()

additional_monthly_conversions = monthly_users * abs_lift
monthly_revenue_lift = additional_monthly_conversions * avg_order_val
annual_revenue_lift = monthly_revenue_lift * 12

if p_val < 0.05 and p_val_aov > 0.05:
    print("DECISION: LAUNCH VARIANT ")
    print(f"Reason: Statistically significant conversion lift (+{rel_lift:.2%}) with no negative impact on AOV.")
    print(f"Projected Annual Revenue Lift: ${annual_revenue_lift:,.2f}")
else:
    print("DECISION: DO NOT LAUNCH / ITERATE ")
    print("Reason: Lift is either statistically insignificant or harms Average Order Value.")
