import pandas as pd
import matplotlib.pyplot as plt

# ---------- Đọc dữ liệu ----------
df = pd.read_csv("Data.csv")


def find_col(keywords):
    """Tự động dò tên cột theo từ khóa (không phụ thuộc tên cột chính xác)."""
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    return None


height_col = find_col(["cao", "height"])
weight_col = find_col(["nang", "nặng", "weight", "kg"])

print("Các cột trong file:", list(df.columns))
print(f"Cột chiều cao: {height_col} | Cột cân nặng: {weight_col}")
print("\n5 dòng đầu:")
print(df.head())
print("\nThống kê mô tả:")
print(df[[height_col, weight_col]].describe())

# ---------- Vẽ scatter plot ----------
plt.figure(figsize=(8, 5))
plt.scatter(df[height_col], df[weight_col], color="blue", s=60, edgecolors="k")
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.title(f"Dữ liệu gốc: {len(df)} người")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("01_data_visualization.png", dpi=120)
print("\nĐã lưu biểu đồ: 01_data_visualization.png")
