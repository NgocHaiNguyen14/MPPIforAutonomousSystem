import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
data = pd.read_csv('./MH91_polar.csv')


# Clean column names
data.columns = data.columns.str.strip()
alpha = data['alpha'].values.reshape(-1, 1)
CL = data['CL'].values
CD = data['CD'].values

# Generate polynomial features (degree 3)
poly = PolynomialFeatures(degree=3)
alpha_poly = poly.fit_transform(alpha)  # adds alpha^2 term

# Fit quadratic model for CL
reg_CL = LinearRegression().fit(alpha_poly, CL)
CL_pred = reg_CL.predict(alpha_poly)

# Fit quadratic model for CD
reg_CD = LinearRegression().fit(alpha_poly, CD)
CD_pred = reg_CD.predict(alpha_poly)

# Print model coefficients
a1, b1, c1, d1 = reg_CL.coef_[3], reg_CL.coef_[2], reg_CL.coef_[1], reg_CL.intercept_
a2, b2, c2, d2 = reg_CD.coef_[3], reg_CD.coef_[2], reg_CD.coef_[1], reg_CD.intercept_
print(f"CL = {a1:.4f}*alpha^3 + {b1:.4f}*alpha^2 + {c2:.4f}*alpha + {d1:.4f}")
print(f"CD = {a2:.4f}*alpha^3 + {b2:.4f}*alpha^2 + {c2:.4f}*alpha + {d2:.4f}")

# Plot CL
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(alpha, CL, label='Data', color='blue')
plt.plot(alpha, CL_pred, label='Quadratic Fit', color='red')
plt.xlabel('Alpha (deg)')
plt.ylabel('C_L')
plt.title('Quadratic Fit: C_L vs Alpha')
plt.legend()

# Plot CD
plt.subplot(1, 2, 2)
plt.scatter(alpha, CD, label='Data', color='green')
plt.plot(alpha, CD_pred, label='Quadratic Fit', color='red')
plt.xlabel('Alpha (deg)')
plt.ylabel('C_D')
plt.title('Quadratic Fit: C_D vs Alpha')
plt.legend()

plt.tight_layout()
plt.show()

