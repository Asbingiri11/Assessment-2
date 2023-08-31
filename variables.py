import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("po1_data.txt")
data.columns = ["Subject identifier","Jitter-%","Jitter-Microseconds",
              "Jitter-r.a.p","Jitter-p.p.q.5",
              "Jitter-d.d.p","Shimmer-%","Shimmer-DB","Shimmer-a.p.q.3",
              "Shimmer-a.p.q.5","Shimmer-a.p.q.11","Shimmer-d.d.a",
              "Harmonicity-autocorrelation","Harmonicity-Harmonic to noise"
              ,"Harmonicity-Harmonic to Noise", "Pitch-Median", "Pitch-Mean"
              ,"Pitch-S.D","Pitch-Minimum","Pitch-Maximum","Pulse-Pulses",
              "Pulse-Periods","Pulse-Mean","Pulse-S.D","Voice-Fraction","Voice-Number","Voice-Degree",
              "UPDRS","PD indicator"
             ]
data1 = data

# Display basic information about the dataset
print("DATA EXPLORATION\n")
print(f"Dataset contains {data1.shape[0]} samples and {data1.shape[1]} features.")
print("-------------------------------------------------------------")
print(data1.info())
print("-------------------------------------------------------------\n")

# Display summary statistics of the dataset
print("SUMMARY STATISTICS\n")
print(data1.describe())
print("-------------------------------------------------------------\n")

# Visualize data distributions by PD indicator
selected_features = ["Jitter-%", "Shimmer-%", "Pulse-Mean", "Voice-Fraction"]
plt.figure(figsize=(12, 8))
for feature in selected_features:
    plt.subplot(2, 2, selected_features.index(feature) + 1)
    sns.histplot(data=data1, x=feature, hue="PD indicator", element="step", common_norm=False)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Data Wrangling
X = data1[selected_features]
y = data1["PD indicator"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nDATA PREPARATION FOR MODELING\n")
print("Data has been split into training and testing sets.")
