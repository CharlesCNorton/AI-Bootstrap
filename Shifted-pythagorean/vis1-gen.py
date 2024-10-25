import numpy as np
import matplotlib.pyplot as plt

# Extended dataset for family sizes and errors from the comprehensive empirical analysis
family_sizes_extracted = [
    95, 79, 71, 63, 59, 55, 53, 47, 44, 41, 39, 35, 31, 29, 27, 26, 24, 23, 21, 20
]
errors_extracted = [
    0.00652520, 0.00850603, 0.00765893, 0.01589721, 0.01171182, 0.00239186, 0.00960759,
    0.01748527, 0.00595247, 0.00231844, 0.02119881, 0.01504894, 0.02843702, 0.02400468,
    0.02923668, 0.01966397, 0.00593402, 0.03116594, 0.03367652, 0.04566806
]

# Creating a scatter plot to show the error from sqrt(2) by family size
plt.figure(figsize=(12, 8))
plt.scatter(family_sizes_extracted, errors_extracted, color='blue', label='Error from $\sqrt{2}$')

# Fit a linear regression line to the data
coefficients = np.polyfit(family_sizes_extracted, errors_extracted, 1)  # Linear regression (degree 1)
polynomial = np.poly1d(coefficients)
trendline_values = polynomial(family_sizes_extracted)

# Plotting the regression trendline
plt.plot(family_sizes_extracted, trendline_values, color='red', linestyle='--', linewidth=2, label='Trendline (Linear Regression)')

# Adding labels and title with increased font size for clarity
plt.xlabel('Family Size', fontsize=18)
plt.ylabel(r'Error from $\sqrt{2}$', fontsize=18)
plt.title(r'Error from $\sqrt{2}$ vs. Family Size with Trendline', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)

# Showing the updated scatter plot with the regression trendline
plt.show()

# Adding additional data for reference - Largest Families and Specific Z-values
largest_families_z_values = [
    330182, 617427, 652082, 700107, 780262, 819668, 899168, 920418, 946343
]
largest_families_ratios = [
    1.40925776, 1.40823523, 1.40218602, 1.41226989, 1.41004210, 1.39798023,
    1.41285325, 1.40430255, 1.41206821
]
errors_for_largest_families = [
    0.00495580, 0.00597833, 0.01202754, 0.00194367, 0.00417146,
    0.01623334, 0.00136031, 0.00991101, 0.00214536
]

# Creating a separate scatter plot for largest families and their convergence behavior
plt.figure(figsize=(12, 8))
plt.scatter(largest_families_z_values, errors_for_largest_families, color='green', label='Error from $\sqrt{2}$ for Largest Families')

# Fit a linear regression line to this data as well
coefficients_large = np.polyfit(largest_families_z_values, errors_for_largest_families, 1)
polynomial_large = np.poly1d(coefficients_large)
trendline_large_values = polynomial_large(largest_families_z_values)

# Plotting the regression trendline for largest families
plt.plot(largest_families_z_values, trendline_large_values, color='orange', linestyle='--', linewidth=2, label='Trendline (Largest Families)')

# Adding labels and title with increased font size for clarity
plt.xlabel('Z Value', fontsize=18)
plt.ylabel(r'Error from $\sqrt{2}$', fontsize=18)
plt.title(r'Error from $\sqrt{2}$ for Largest Families vs. Z Value with Trendline', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)

# Showing the updated scatter plot for largest families
plt.show()
