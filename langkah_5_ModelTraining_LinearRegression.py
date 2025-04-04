import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df_outliers = pd.read_csv("dataset_with_outliers.csv")
df_standard_scaled = pd.read_csv("dataset_scaled_standard.csv")
df_minmax_scaled = pd.read_csv("dataset_scaled_MinMaxScaler.csv")

# Pastikan semua dataset memiliki kategori yang sama
all_dfs = [df_outliers, df_standard_scaled, df_minmax_scaled]
label_encoders = {}

for col in df_outliers.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    all_values = pd.concat([df[col].dropna() for df in all_dfs if col in df.columns], axis=0)
    label_encoders[col].fit(all_values)
    
    for df in all_dfs:
        if col in df.columns:
            df[col] = df[col].fillna(all_values.mode()[0])
            df[col] = label_encoders[col].transform(df[col])

# Handle missing values
def handle_missing_values(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(inplace=True)
    return df

for i in range(len(all_dfs)):
    all_dfs[i] = handle_missing_values(all_dfs[i])

# Pilih fitur dan target
X_outliers, y_outliers = df_outliers.iloc[:, :-1], df_outliers.iloc[:, -1]
X_standard, y_standard = df_standard_scaled.iloc[:, :-1], df_standard_scaled.iloc[:, -1]
X_minmax, y_minmax = df_minmax_scaled.iloc[:, :-1], df_minmax_scaled.iloc[:, -1]

def handle_nan_target(y):
    return y.fillna(y.mode()[0]) if y.dtype == 'O' else y.fillna(y.median())

y_outliers = handle_nan_target(y_outliers)
y_standard = handle_nan_target(y_standard)
y_minmax = handle_nan_target(y_minmax)

# Split data
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_outliers, y_outliers, test_size=0.2, random_state=42)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_standard, y_standard, test_size=0.2, random_state=42)
X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(X_minmax, y_minmax, test_size=0.2, random_state=42)

# Fungsi untuk melatih model dan menyimpan hasil visualisasi
def train_evaluate_model(X_train, X_test, y_train, y_test, title, filename):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{title}:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}\n")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scatter plot Prediksi vs Aktual
    axes[0].scatter(y_test, y_pred, alpha=0.6, label="Prediksi")
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='dashed', label="Garis Ideal")
    axes[0].set_xlabel("Nilai Aktual")
    axes[0].set_ylabel("Prediksi")
    axes[0].set_title(f"{title} - Prediksi vs Aktual")
    axes[0].legend()
    
    # Residual plot
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1])
    axes[1].set_xlabel("Residual")
    axes[1].set_title(f"{title} - Distribusi Residual")
    
    # Residual vs Prediksi
    axes[2].scatter(y_pred, residuals, alpha=0.6)
    axes[2].axhline(y=0, color='r', linestyle='--')
    axes[2].set_xlabel("Prediksi")
    axes[2].set_ylabel("Residual")
    axes[2].set_title(f"{title} - Residual Plot")
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return mse, r2

# Latih model dan simpan hasil visualisasi
mse_out, r2_out = train_evaluate_model(X_train_out, X_test_out, y_train_out, y_test_out, 
                                       "Dataset dengan Outlier", "visualisasi_outlier")
mse_std, r2_std = train_evaluate_model(X_train_std, X_test_std, y_train_std, y_test_std, 
                                       "Dataset Standard Scaled", "visualisasi_standard_scaled")
mse_min, r2_min = train_evaluate_model(X_train_min, X_test_min, y_train_min, y_test_min, 
                                       "Dataset MinMax Scaled", "visualisasi_minmax_scaled")

# Tampilkan hasil perbandingan
result_df = pd.DataFrame({
    "Dataset": ["Dengan Outlier", "Standard Scaled", "MinMax Scaled"],
    "MSE": [mse_out, mse_std, mse_min],
    "R2 Score": [r2_out, r2_std, r2_min]
})

print("\nPerbandingan Hasil Model:")
print(result_df)
