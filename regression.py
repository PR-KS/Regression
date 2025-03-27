import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

df = pd.read_csv("gold_price_regression.csv")

print(df.head())
print(df.info())

target_column = "gold close"
if target_column not in df.columns:
    raise ValueError("Целевая переменная не найдена в датасете!")

df = df.dropna(subset=[target_column])

df = df.drop(columns=["date"], errors="ignore")

X = df.drop(columns=[target_column])
y = df[target_column]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}\n")

plt.scatter(y_test, models["Random Forest"].predict(X_test_scaled), alpha=0.5)
plt.xlabel("Фактическая цена золота")
plt.ylabel("Предсказанная цена")
plt.title("Сравнение предсказаний Random Forest")
plt.show()
