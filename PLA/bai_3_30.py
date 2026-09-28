# Bài 3.30: Perceptron phân lớp nhị phân - tập Breast Cancer (sklearn)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = make_pipeline(StandardScaler(), Perceptron(random_state=42))
grid = {"perceptron__alpha": [0, 1e-4, 1e-3, 1e-2],
        "perceptron__penalty": [None, "l2"],
        "perceptron__max_iter": [100, 1000]}
model = GridSearchCV(pipe, grid, scoring="f1", cv=5).fit(X_tr, y_tr)

p = model.predict(X_te)
print("Tham số tốt nhất:", model.best_params_)
print(f"Accuracy : {accuracy_score(y_te, p):.4f}")
print(f"Precision: {precision_score(y_te, p):.4f}")
print(f"Recall   : {recall_score(y_te, p):.4f}")
print(f"F1-score : {f1_score(y_te, p):.4f}")
