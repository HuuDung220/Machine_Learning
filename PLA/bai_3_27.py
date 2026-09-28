# Bài 3.27: Dự đoán bằng Perceptron
import numpy as np

w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y = -1

s = w @ x
pred = 1 if s >= 0 else -1
print("w^T x =", s)
print("Nhãn dự đoán =", pred)
print("Bị phân lớp sai?", pred != y)
