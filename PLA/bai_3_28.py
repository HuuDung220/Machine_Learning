# Bài 3.28: Một bước cập nhật Perceptron
import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1

s = w @ x
print("w^T x =", s, "-> sai?", np.sign(s) != y)
if np.sign(s) != y:
    w = w + y * x
    print("w mới =", w)
    print("w^T x sau cập nhật =", w @ x)
