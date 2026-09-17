import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ---------- Đọc dữ liệu ----------
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
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
degrees = list(range(1, 13))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for d in degrees:
    pipeline = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )
    scores = cross_val_score(
        pipeline, X_train_full, y_train_full, cv=kf, scoring="r2"
    )
    cv_scores.append(scores.mean())
    print(f"Degree {d:2d}: CV R2 trung bình = {scores.mean():.4f}")

best_degree = degrees[int(np.argmax(cv_scores))]
print(f"\n=> Degree tối ưu theo CV: {best_degree}")

# ---------- Vẽ đường cong CV score theo degree ----------
plt.figure(figsize=(8, 5))
plt.plot(degrees, cv_scores, marker="o")
plt.axvline(best_degree, color="red", linestyle="--",
            label=f"Degree tốt nhất = {best_degree}")
plt.xlabel("Bậc đa thức (degree)")
plt.ylabel("CV R2 trung bình (5-Fold)")
plt.title("Chọn độ phức tạp mô hình bằng K-Fold Cross Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("03_cv_degree_selection.png", dpi=120)
print("Đã lưu biểu đồ: 03_cv_degree_selection.png")
final_model = make_pipeline(
    PolynomialFeatures(degree=best_degree, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)
final_model.fit(X_train_full, y_train_full)

y_train_pred = final_model.predict(X_train_full)
y_test_pred = final_model.predict(X_test)

train_r2 = r2_score(y_train_full, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train_full, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"\nMô hình cuối cùng (degree={best_degree}):")
print(f"Train R2: {train_r2:.4f} | Train MSE: {train_mse:.2f}")
print(f"Test  R2: {test_r2:.4f} | Test  MSE: {test_mse:.2f}")
print("=> Khoảng cách Train/Test R2 nhỏ hơn nhiều so với mô hình overfit ở bước 2")
x_min, x_max = X.min() - 5, X.max() + 5
x_range = np.linspace(x_min, x_max, 300).reshape(-1, 1)
y_range_pred = final_model.predict(x_range)

plt.figure(figsize=(8, 5))
plt.scatter(X_train_full, y_train_full, color="blue", s=60, label="Train", zorder=3)
plt.scatter(X_test, y_test, color="green", s=60, label="Test", zorder=3)
plt.plot(x_range, y_range_pred, color="darkorange", linewidth=2,
          label=f"Mô hình tối ưu (degree={best_degree})")
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.title(f"SAU KHI DÙNG K-FOLD CV — Train R2={train_r2:.3f} | Test R2={test_r2:.3f}")
plt.ylim(y.min() - 15, y.max() + 15)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("03_final_model_result.png", dpi=120)
print("Đã lưu biểu đồ: 03_final_model_result.png")
