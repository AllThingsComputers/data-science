# Install required packages
!pip install pandas numpy scipy scikit-learn matplotlib

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# Update plot parameters
plt.rcParams.update({'font.size': 14})

# Load the dataset
# Replace 'anomaly_sec_events.csv' with the full path to your dataset if necessary
df = pd.read_csv('anomaly_sec_events.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Extract features
# Assuming 'military_time' and 'EventID' exist in the dataset
# If there are other features you'd like to use, adjust this section
Account = df["Account"]
data = df.loc[:, ["military_time", "EventID"]]

# Display basic statistics
print("\nDataset Statistics:")
print(data.describe())

# Define a parameter to analyze (replace 'EventID' if needed)
param = "EventID"

# Calculate quartiles
qv1 = data[param].quantile(0.25)
qv2 = data[param].quantile(0.5)
qv3 = data[param].quantile(0.75)

# Calculate interquartile range and anomaly threshold
iqr = qv3 - qv1
lower_limit = qv1 - 1.5 * iqr
upper_limit = qv3 + 1.5 * iqr

print(f"\nAnomaly Thresholds for {param}:")
print(f"Lower limit: {lower_limit}")
print(f"Upper limit: {upper_limit}")

# Filter anomalies based on IQR
anomalies_iqr = data[(data[param] < lower_limit) | (data[param] > upper_limit)]
print(f"\nDetected {len(anomalies_iqr)} anomalies using IQR:")

# Display anomalies
print(anomalies_iqr)

# Visualize anomalies using box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data[param], vert=False, patch_artist=True)
plt.title(f"Boxplot of {param}")
plt.xlabel(param)
plt.grid()
plt.show()

# Unsupervised anomaly detection using OneClassSVM
print("\nRunning OneClassSVM anomaly detection...")
one_class_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
data_scaled = (data - data.mean()) / data.std()  # Standardize features
one_class_svm.fit(data_scaled)

# Predict anomalies
data['Anomaly_SVM'] = one_class_svm.predict(data_scaled)
anomalies_svm = data[data['Anomaly_SVM'] == -1]
print(f"\nDetected {len(anomalies_svm)} anomalies using OneClassSVM:")
print(anomalies_svm)

# Visualize anomalies using scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data[param], label='Normal', color='blue')
plt.scatter(anomalies_svm.index, anomalies_svm[param], label='Anomalies', color='red')
plt.title(f"OneClassSVM Anomaly Detection: {param}")
plt.xlabel("Index")
plt.ylabel(param)
plt.legend()
plt.grid()
plt.show()

# Anomaly detection using EllipticEnvelope
print("\nRunning EllipticEnvelope anomaly detection...")
elliptic_envelope = EllipticEnvelope(contamination=0.05)
elliptic_envelope.fit(data_scaled)

# Predict anomalies
data['Anomaly_EE'] = elliptic_envelope.predict(data_scaled)
anomalies_ee = data[data['Anomaly_EE'] == -1]
print(f"\nDetected {len(anomalies_ee)} anomalies using EllipticEnvelope:")
print(anomalies_ee)

# Visualize EllipticEnvelope results
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data[param], label='Normal', color='blue')
plt.scatter(anomalies_ee.index, anomalies_ee[param], label='Anomalies', color='orange')
plt.title(f"EllipticEnvelope Anomaly Detection: {param}")
plt.xlabel("Index")
plt.ylabel(param)
plt.legend()
plt.grid()
plt.show()
