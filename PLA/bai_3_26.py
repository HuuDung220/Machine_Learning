# Bài 3.26: Gradient Descent cho f(x) = x^2 - 4x + 5
f = lambda x: x**2 - 4*x + 5
df = lambda x: 2*x - 4          # f'(x) = 2x - 4

x, eta = 5.0, 0.2
print(f"x0 = {x}, f = {f(x)}")
for k in range(1, 5):
    x = x - eta * df(x)
    print(f"x{k} = {x:.4f}, f = {f(x):.4f}")
# Nhận xét: x -> 2 (nghiệm x* = 2, f_min = 1); mỗi bước (x-2) giảm còn 0.6 lần => hội tụ
