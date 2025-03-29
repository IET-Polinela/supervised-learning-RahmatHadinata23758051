import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("train.csv") 

# 1 Encoding Fitur Kategorikal
categorical_features = ['Neighborhood', 'HouseStyle']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

# 2 Memilih Fitur X dan Target Y
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF'] 
X = df[selected_features]
Y = df['SalePrice']

# 3 Mengatasi Missing Values
X = X.copy()  # Hindari SettingWithCopyWarning
X.fillna(X.median(), inplace=True)  # Isi nilai kosong dengan median

# 4 Membagi Data menjadi Training & Testing Set (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Training Set:", X_train.shape, Y_train.shape)
print("Testing Set:", X_test.shape, Y_test.shape)
