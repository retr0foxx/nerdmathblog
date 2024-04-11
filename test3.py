import matplotlib.pyplot as plt
import numpy as np

# Define the linear function: y = mx + b
def linear_function(x, m=2, b=1):
    return m * x + b

# Generate random data points around the linear function with noise
np.random.seed(0)  # Set random seed for reproducibility

N = 100;
# Generate x values
x_values = np.linspace(0, 10, 100)

# Calculate corresponding y values based on the linear function
y_values_actual = linear_function(x_values)

# Add random noise to the y values
noise = np.random.normal(loc=0, scale=1, size=len(x_values))
y_values_noisy = y_values_actual + noise

with open("text3.txt", "w") as f:
    f.write("[");
    for i in range(len(y_values_noisy)):
        f.write(str(y_values_noisy[i]));
        if (i + 1 < len(y_values_noisy)):
            f.write(",");

    f.write("]");

sum_y = np.sum(y_values_noisy)
sum_x = np.sum(x_values);

nom = N * np.dot(x_values, y_values_noisy) - np.sum(x_values * sum_y);
denom = N * np.dot(x_values, x_values - sum_x);
print(nom / denom, nom, denom);

print(np.sum(x_values * (N * y_values_noisy - np.sum(y_values_noisy))), N * np.sum(x_values * (x_values - np.sum(x_values))));
print()
print(np.dot(x_values, y_values_noisy - 1) / np.dot(x_values, x_values), np.sum(y_values_noisy - 2 * x_values) / N)
print(np.sum(N * x_values * y_values_noisy - x_values * np.sum(y_values_noisy - 2 * x_values)) / (N * np.dot(x_values, x_values)))
print(np.sum(N * x_values * y_values_noisy - x_values * np.sum(y_values_noisy) + 2 * x_values * np.sum(x_values)) / (N * np.dot(x_values, x_values)))
print((np.sum(N * x_values * y_values_noisy - x_values * np.sum(y_values_noisy)) + 2 * np.sum(x_values * np.sum(x_values))) / (N * np.dot(x_values, x_values)))

print(2 * N * np.dot(x_values, x_values) - 2 * np.sum(x_values * np.sum(x_values)), '=', np.sum(x_values * (N * y_values_noisy - np.sum(y_values_noisy))));
print(2, '=', np.sum(x_values * (N * y_values_noisy - np.sum(y_values_noisy))) / (N * np.dot(x_values, x_values) - np.sum(x_values * np.sum(x_values))));
print("denom evoluation:");
print(N * np.dot(x_values, x_values) - np.sum(x_values * np.sum(x_values)))
print(N * np.sum(x_values * (x_values - np.sum(x_values))))
print(N * np.dot(x_values, x_values - np.sum(x_values)))
print(N * np.dot(x_values, x_values) - np.sum(x_values * np.sum(x_values)));

#[x_1, x_2, ...] \cdot [x_1 - c, x_2 - c, ...] = x_1(x_1 - c) + x_2(x_2 - c) + ...
#                                              = x_1x_1 + x_2x_2 + ... - x_1c - x_2c

# Plot the linear function and the noisy data points
plt.figure(figsize=(8, 6))
# plt.plot(x_values, y_values_actual, label='Actual Function', color='blue')
plt.plot(x_values, linear_function(x_values, m=nom / denom))
plt.scatter(x_values, y_values_noisy, label='Noisy Data Points', color='red', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Points Based on Linear Function with Noise')
plt.legend()
plt.grid(True)
plt.show()
