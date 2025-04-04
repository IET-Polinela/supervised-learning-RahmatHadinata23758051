import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset tanpa outlier
df_no_outliers = pd.read_csv("dataset_without_outliers.csv")

# Pilih fitur dan target (asumsi kolom terakhir adalah target)
X = df_no_outliers.iloc[:, :-1]
y = df_no_outliers.iloc[:, -1]

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fungsi untuk melatih Polynomial Regression, evaluasi performa, dan cek overfitting
def evaluate_poly_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Polynomial Degree {degree}:")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  R² Training: {train_r2:.4f}, R² Testing: {test_r2:.4f}\n")

    # Visualisasi Prediksi vs Aktual
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_test_pred, alpha=0.6, label="Prediksi")
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='dashed', label="Garis Ideal")
    plt.xlabel("Nilai Aktual")
    plt.ylabel("Prediksi")
    plt.title(f"Polynomial Regression Degree {degree} - Prediksi vs Aktual")
    plt.legend()
    plt.savefig(f"polynomial_degree_{degree}_prediction_vs_actual.png")  # Simpan gambar
    plt.show()

    return model, poly

# Evaluasi untuk degree 2 dan 3
evaluate_poly_model(2)
evaluate_poly_model(3)

# ===== Learning Curve =====
def plot_learning_curve(degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()

    train_sizes, train_scores, test_scores = learning_curve(model, X_poly, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Error')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Testing Error')
    plt.title(f'Learning Curve for Polynomial Degree {degree}')
    plt.xlabel('Training Size')
    plt.ylabel('Error (MSE)')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(f"learning_curve_degree_{degree}.png")  # Simpan gambar
    plt.show()

# Plot learning curve untuk degree 2 dan 3
plot_learning_curve(2)
plot_learning_curve(3)
