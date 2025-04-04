import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset (ganti dengan dataset yang digunakan)
df = pd.read_csv("train.csv")

# Hapus kolom yang semuanya kosong
df.dropna(axis=1, how='all', inplace=True)

# Pisahkan fitur numerik & kategorikal
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Encoding kolom kategorikal
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = df[col].astype(str).fillna("Missing")  # Isi NaN dengan "Missing"
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Hapus missing values pada data numerik
df.fillna(df.median(numeric_only=True), inplace=True)

# Simpan dataset dengan outlier (tapi udah bersih dari kategori & NaN)
df.to_csv("dataset_with_outliers.csv", index=False)

# Identifikasi & hapus outlier dengan metode IQR
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

df_without_outliers = remove_outliers_iqr(df, numeric_features)

# Simpan dataset tanpa outlier
df_without_outliers.to_csv("dataset_without_outliers.csv", index=False)

# Visualisasi Boxplot sebelum & sesudah menangani outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_features])
plt.xticks(rotation=90)
plt.title("Boxplot Sebelum Penanganan Outlier")
plt.savefig("boxplot_sebelum_outlier.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_without_outliers[numeric_features])
plt.xticks(rotation=90)
plt.title("Boxplot Setelah Penanganan Outlier")
plt.savefig("boxplot_setelah_outlier.png", dpi=300, bbox_inches='tight')
plt.close()
