import numpy as np
import pandas as pd

TARGET = "RuiRo"
df = pd.read_csv("data.csv").drop(columns="ID")

# Rời rạc hóa giống bài: 3 nhóm
df["Tuoi"] = pd.cut(df["Tuoi"], bins=[25, 33, 41, 49], right=False,
                    labels=["25-33", "33-41", "41-49"])
df["ThuNhap"] = pd.cut(df["ThuNhap"], bins=[5e6, 10e6, 15e6, 20e6], right=False,
                       labels=["5-10tr", "10-15tr", "15-20tr"])
df = df.astype(str)

def entropy(y):
    p = y.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

def info_gain(d, attr):
    n = len(d)
    rem = sum(len(s) / n * entropy(s[TARGET]) for _, s in d.groupby(attr))
    return entropy(d[TARGET]) - rem

def id3(d, attrs, indent=0):
    pad = "  " * indent
    y = d[TARGET]
    if y.nunique() == 1:
        return y.iloc[0]
    if not attrs:
        return y.mode()[0]
    gains = {a: info_gain(d, a) for a in attrs}
    best = max(gains, key=gains.get)
    cnt = y.value_counts().to_dict()
    print(f"{pad}Entropy(S) = {entropy(y):.4f}  (n={len(d)}, phân bố={cnt})")
    for a, g in gains.items():
        print(f"{pad}  Gain({a}) = {g:.4f}")
    print(f"{pad}=> Chọn: {best}\n")
    tree = {best: {}}
    rest = [a for a in attrs if a != best]
    for val, sub in d.groupby(best):
        print(f"{pad}[{best} = {val}]")
        tree[best][val] = id3(sub, rest, indent + 1)
    return tree

def show(tree, indent=0):
    pad = "    " * indent
    if not isinstance(tree, dict):
        print(f"{pad}→ RuiRo = {tree}")
        return
    attr = next(iter(tree))
    for val, sub in tree[attr].items():
        print(f"{pad}{attr} = {val}")
        show(sub, indent + 1)

attrs = [c for c in df.columns if c != TARGET]
tree = id3(df, attrs)
print("\n===== CÂY QUYẾT ĐỊNH ID3 =====")
show(tree)
