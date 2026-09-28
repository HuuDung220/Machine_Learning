# Bài 3.29: Lớp Perceptron (nhãn -1/+1)
import numpy as np

class Perceptron:
    def __init__(self, lr=1.0, max_iter=100):
        self.lr, self.max_iter = lr, max_iter

    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), X]          # thêm bias
        self.w = np.zeros(X.shape[1])
        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (self.w @ xi) <= 0:    # phân lớp sai
                    self.w += self.lr * yi * xi
                    errors += 1
            if errors == 0:                    # hội tụ
                break
        return self

    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        return np.where(X @ self.w >= 0, 1, -1)

if __name__ == "__main__":
    X = np.array([[1, 1], [2, 2], [3, 1], [-1, -1], [-2, -1], [-3, -2]])
    y = np.array([1, 1, 1, -1, -1, -1])
    model = Perceptron().fit(X, y)
    print("w =", model.w)
    print("Dự báo:", model.predict(np.array([[2, 3], [-2, -3]])))
