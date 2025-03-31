import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset (disesuaikan dengan dataset tanpa outlier)
df = pd.read_csv("dataset_without_outliers.csv")  

# Pilih hanya kolom numerik untuk dilakukan scaling
numerical_cols = df.select_dtypes(include=[np.number]).columns

# **Perbaikan untuk menangani NaN dan Inf**
df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)  # Ganti inf dengan NaN
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())  # Isi NaN dengan median

# Lakukan Scaling
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)
df_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

# Tentukan metode scaling terbaik berdasarkan distribusi data
skewness = df[numerical_cols].skew().mean()
kurtosis = df[numerical_cols].kurtosis().mean()

if abs(skewness) < 1 and abs(kurtosis - 3) < 1:  
    best_scaler = "StandardScaler"
    df_best_scaled = df_standard_scaled
else:
    best_scaler = "MinMaxScaler"
    df_best_scaled = df_minmax_scaled

# Simpan dataset dengan metode terbaik
output_file = f"dataset_scaled_{best_scaler}.csv"
df_best_scaled.to_csv(output_file, index=False)

# Plot histogram sebelum dan sesudah scaling
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram data asli
sns.histplot(df[numerical_cols].melt(value_name="Value")["Value"], bins=50, kde=True, ax=axes[0])
axes[0].set_title("Distribusi Data Asli")

# Histogram setelah StandardScaler
sns.histplot(df_standard_scaled.melt(value_name="Value")["Value"], bins=50, kde=True, ax=axes[1])
axes[1].set_title("Distribusi Setelah StandardScaler")

# Histogram setelah MinMaxScaler
sns.histplot(df_minmax_scaled.melt(value_name="Value")["Value"], bins=50, kde=True, ax=axes[2])
axes[2].set_title("Distribusi Setelah MinMaxScaler")

plt.tight_layout()

# Simpan hasil visualisasi
plt.savefig("hasil_scaling_histogram.png", dpi=300)
plt.show()

# Print metode yang dipilih
print(f"Metode Scaling yang dipilih: {best_scaler}")
print(f"Dataset hasil scaling disimpan sebagai: {output_file}")
