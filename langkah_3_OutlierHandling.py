import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (ganti dengan dataset yang digunakan)
df = pd.read_csv("train.csv")

# Pilih fitur numerik untuk analisis outlier
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Visualisasi Boxplot sebelum menangani outlier dan simpan gambar
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_features])
plt.xticks(rotation=90)
plt.title("Boxplot Sebelum Penanganan Outlier")
plt.savefig("boxplot_sebelum_outlier.png", dpi=300, bbox_inches='tight')
plt.close()

# Identifikasi outlier menggunakan metode IQR
def remove_outliers_iqr(df, features):
    df_clean = df.copy()
    for feature in features:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
    return df_clean

# Buat dua dataset: dengan outlier & tanpa outlier
df_with_outliers = df.copy()
df_without_outliers = remove_outliers_iqr(df, numeric_features)

# Visualisasi Boxplot setelah menangani outlier dan simpan gambar
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_without_outliers[numeric_features])
plt.xticks(rotation=90)
plt.title("Boxplot Setelah Penanganan Outlier")
plt.savefig("boxplot_setelah_outlier.png", dpi=300, bbox_inches='tight')
plt.close()

# Simpan hasil dataset
df_with_outliers.to_csv("dataset_with_outliers.csv", index=False)
df_without_outliers.to_csv("dataset_without_outliers.csv", index=False)
