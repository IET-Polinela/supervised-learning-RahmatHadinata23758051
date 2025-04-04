import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Load dataset tanpa outlier
df = pd.read_csv("dataset_without_outliers.csv")

# Encoding kolom kategorikal sebelum scaling
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna("Unknown")  # Isi NaN dengan string "Unknown"
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Pilih hanya kolom numerik untuk dilakukan scaling
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Perbaikan untuk menangani NaN dan Inf
df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)  # Ganti inf dengan NaN
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())  # Isi NaN dengan median

# Hapus kolom yang masih memiliki NaN setelah pemrosesan jika ada
df = df.dropna(axis=1, how='all')

# Perbarui daftar kolom numerik setelah pemrosesan
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Lakukan Scaling
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)
df_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

# Simpan kedua dataset hasil scaling
df_standard_scaled.to_csv("dataset_standard_scaled.csv", index=False)
df_minmax_scaled.to_csv("dataset_minmax_scaled.csv", index=False)

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

# Print informasi
print("Dataset hasil StandardScaler disimpan sebagai: dataset_standard_scaled.csv")
print("Dataset hasil MinMaxScaler disimpan sebagai: dataset_minmax_scaled.csv")
