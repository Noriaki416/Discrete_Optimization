# Discrete Optimization
## Week 6 Homework (Warehouse Location Problem)


Qiita向けの制約条件3がないモデルの計算を実行するには以下を入力して下さい。
dataset_nameにはdataフォルダにあるデータセットの名前を入力して下さい。

```
python solver_test.py 'data/dataset_name'
```


# 定式化

### 変数

- $x_w$ : 倉庫wがopenすれば1、そうでなければ0。
- $y_{w, c}$ : 客cが倉庫wに割り当てられれば1, そうでなければ0。

### 係数

- $c_w$ : 倉庫wをopenさせるコスト。
- $t_{w, c}$ : 客cと倉庫wとの距離。　
- $Cap_w$ : 倉庫wのキャパ。
- $d_c$ : 客cの需用量。

## 目的項

```math
\begin{align}
\sum_{w}c_w x_w + \sum_{w, c}t_{w, c}y_{w, c}
\end{align}
```

## 制約項

### 1. 客cはどこか1つの倉庫に割り当てられる必要がある。

```math
\begin{align}
\sum_{w}y_{w, c} = 1
\end{align}
```

### 2. 倉庫wがopenしていないと客cを倉庫wに割り当てられない。

```math
\begin{align}
y_{w, c} \le x_{w}
\end{align}
```

### 3. 倉庫のキャパを超えて客を割り当ててはいけない。

```math
\begin{align}
\sum_c d_c y_{w, c} \le Cap_{w}x_{w}
\end{align}
```
