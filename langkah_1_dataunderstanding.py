import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")  # Sesuaikan dengan nama file dataset kamu

# Menampilkan statistik deskriptif untuk semua fitur numerik
stats_numeric = df.describe().T  # Statistik deskriptif dasar
stats_numeric["median"] = df.median(numeric_only=True)  # Menambahkan median
stats_numeric["q1"] = df.quantile(0.25, numeric_only=True)  # Quartile 1 (Q1)
stats_numeric["q3"] = df.quantile(0.75, numeric_only=True)  # Quartile 3 (Q3)
stats_numeric["missing_values"] = df.isnull().sum()  # Jumlah nilai yang hilang

# Menyusun format statistik deskriptif yang sesuai
stats_numeric = stats_numeric[["mean", "median", "std", "min", "q1", "50%", "q3", "max", "count", "missing_values"]]

# Simpan ke CSV
output_file = "langkah_1_statistik_deskriptif.csv"
stats_numeric.to_csv(output_file)

print(f"File statistik deskriptif telah disimpan sebagai {output_file}")

# Identifikasi fitur dengan banyak missing values (> 30% dari total data)
missing_threshold = 0.3 * len(df)  # Ambang batas 30% missing values
cols_to_drop = stats_numeric[stats_numeric["missing_values"] > missing_threshold].index.tolist()
print("\nKolom yang akan dihapus karena terlalu banyak missing values:", cols_to_drop)

# Menghapus kolom yang memiliki banyak missing values
df_cleaned = df.drop(columns=cols_to_drop)

# Mengisi missing values untuk fitur numerik dengan median
for col in df_cleaned.select_dtypes(include=['number']).columns:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# Mengisi missing values untuk fitur kategorikal dengan modus (nilai terbanyak)
for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

# Memeriksa apakah masih ada missing values
print("\nMissing Values Setelah Penanganan:", df_cleaned.isnull().sum().sum())  # Jika 0, berarti sudah bersih
