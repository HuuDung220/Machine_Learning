import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("Data.csv")
def find_col(keywords):
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    return None


height_col = find_col(["cao", "height"])
weight_col = find_col(["nang", "nặng", "weight", "kg"])

X = df[[height_col]].values
y = df[weight_col].values

# ---------- Chia train/test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------- Tạo đặc trưng đa thức bậc cao ----------
DEGREE = 12  # cố ý chọn rất cao so với 10 mẫu train -> overfitting
poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

print(f"Số mẫu train: {len(X_train)} | Số đặc trưng sau đa thức: {X_train_poly.shape[1]}")

# ---------- Huấn luyện (không regularization) ----------
model = LinearRegression()
model.fit(X_train_poly, y_train)

# ---------- Đánh giá ----------
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"\nTrain R2: {train_r2:.4f} | Train MSE: {train_mse:.2f}")
print(f"Test  R2: {test_r2:.4f} | Test  MSE: {test_mse:.2f}")
print("\n=> Train R2 gần 1.0 nhưng Test R2 thấp/âm => mô hình đang OVERFITTING")

# ---------- Vẽ đường cong mô hình ----------
x_min, x_max = X.min() - 5, X.max() + 5
x_range = np.linspace(x_min, x_max, 300).reshape(-1, 1)
x_range_poly = scaler.transform(poly.transform(x_range))
y_range_pred = model.predict(x_range_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="blue", s=60, label="Train", zorder=3)
plt.scatter(X_test, y_test, color="green", s=60, label="Test", zorder=3)
plt.plot(x_range, y_range_pred, color="red", linewidth=2,
          label=f"Mô hình overfit (degree={DEGREE})")
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.title(f"OVERFITTING — Train R2={train_r2:.3f} | Test R2={test_r2:.3f}")
plt.ylim(y.min() - 15, y.max() + 15)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("02_overfit_result.png", dpi=120)
print("Đã lưu biểu đồ: 02_overfit_result.png")
