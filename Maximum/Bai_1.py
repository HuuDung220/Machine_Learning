def cost(x):
    return x*x-2
def grad(x):
    return 2*x
def myGD1(x0, eta):
    x = [x0]
    for i in range(100):
        x_new = x[-1]-eta*grad(x[-1])
        if abs(grad(x_new) < 0.001):
            break
        x.append(x_new)
    return (x, i)
(x, it) = myGD1(x0=5, eta=0.1)
print('Nghiệm x =', x[-1])
print('Số vòng lặp:', it)