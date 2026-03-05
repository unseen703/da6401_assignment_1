"""

QUESTION 5: OLS vs Total Least Squares (TLS) with Noisy Features

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("QUESTION 5: OLS vs TLS")
print("=" * 60)

np.random.seed(42)
N = 200

# True features
x1_true = np.random.normal(0, 1, N)
x2_true = np.random.normal(0, 1, N)
y_true = 3 * x1_true - 2 * x2_true

# Add measurement noise
eps1 = np.random.normal(0, 0.5, N)
eps2 = np.random.normal(0, 0.5, N)
eps_y = np.random.normal(0, 0.2, N)

x1 = x1_true + eps1
x2 = x2_true + eps2
y = y_true + eps_y

# -------------------------------------------------------
# (a) OLS
# -------------------------------------------------------
print("\n(a) Ordinary Least Squares (OLS)")
print("-" * 40)

X = np.column_stack([x1, x2])
X_b = np.column_stack([np.ones(N), x1, x2])  # with intercept

beta_ols = np.linalg.lstsq(X_b, y, rcond=None)[0]
print(f"OLS Intercept: {beta_ols[0]:.4f}")
print(f"OLS β1 (true=3):  {beta_ols[1]:.4f}")
print(f"OLS β2 (true=-2): {beta_ols[2]:.4f}")

y_pred_ols = X_b @ beta_ols
mse_ols = mean_squared_error(y, y_pred_ols)
print(f"OLS Training MSE: {mse_ols:.4f}")

# -------------------------------------------------------
# (b) TLS via SVD
# -------------------------------------------------------
print("\n(b)TLS via SVD")
print("-" * 40)

# TLS: augment [X | y], do SVD, use  last right singular vector
Z = np.column_stack([x1, x2, y])  # N x 3 (no intercept in standard TLS)

# Center data
Z_mean = Z.mean(axis=0)
Z_centered = Z - Z_mean

U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
V = Vt.T

# Last column of V corresponds  to smallest singular value
v_last = V[:, -1]

# TLS solution: coefficients from partitioning v_last into [v_x | v_y]
# β_TLS = -v_x / v_y
v_x = v_last[:2]   # components for x1, x2
v_y = v_last[2]    # component for y

beta_tls = -v_x / v_y

# Intercept from means: intercept = mean(y) - beta_tls @ mean([x1, x2])
intercept_tls = Z_mean[2] - beta_tls @ Z_mean[:2]

print(f"TLS Intercept: {intercept_tls:.4f}")
print(f"TLS β1 (true=3):  {beta_tls[0]:.4f}")
print(f"TLS β2 (true=-2): {beta_tls[1]:.4f}")

y_pred_tls = intercept_tls + X @ beta_tls
mse_tls = mean_squared_error(y, y_pred_tls)
print(f"TLS Training MSE: {mse_tls:.4f}")

# -------------------------------------------------------
# (c) Comparison
# -------------------------------------------------------
print("\n(c) Comparison with True Parameters (3, -2)")
print("-" * 40)
true_params = np.array([3.0, -2.0])

ols_error = np.linalg.norm(beta_ols[1:] - true_params)
tls_error = np.linalg.norm(beta_tls - true_params)





print(f"OLS coefficients: β1={beta_ols[1]:.4f}, β2={beta_ols[2]:.4f}")
print(f"TLS coefficients: β1={beta_tls[0]:.4f}, β2={beta_tls[1]:.4f}")
print(f"True parameters:  β1=3.0000, β2=-2.0000")
print(f"\nOLS L2 error from true: {ols_error:.4f}")
print(f"TLS L2 error from true: {tls_error:.4f}")
print(f"\n→ {'TLS' if tls_error < ols_error else 'OLS'} is closer to the true parameters.")
print("  TLS accounts for errors in X (not just y), making it better when")
print("  features are noisy. OLS assumes X is error-free, causing attenuation bias.")

# Plot Q5
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Question 5: OLS vs TLS Coefficient Comparison", fontsize=14, fontweight='bold')

methods = ['OLS', 'TLS', 'True']
b1_vals = [beta_ols[1], beta_tls[0], 3.0]
b2_vals = [beta_ols[2], beta_tls[1], -2.0]
colors = ['steelblue', 'darkorange', 'green']

axes[0].bar(methods, b1_vals, color=colors)
axes[0].axhline(3.0, color='green', linestyle='--', alpha=0.5)
axes[0].set_title("β₁ Estimates (True = 3)")
axes[0].set_ylabel("Coefficient Value")
for i, v in enumerate(b1_vals):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

axes[1].bar(methods, b2_vals, color=colors)
axes[1].axhline(-2.0, color='green', linestyle='--', alpha=0.5)
axes[1].set_title("β₂ Estimates (True = -2)")
axes[1].set_ylabel("Coefficient Value")
for i, v in enumerate(b2_vals):
    axes[1].text(i, v - 0.08, f"{v:.3f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('q5_ols_vs_tls.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Plot saved: q5_ols_vs_tls.png]")

